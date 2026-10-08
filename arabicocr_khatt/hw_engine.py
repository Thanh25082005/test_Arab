# arabicocr_khatt/hw_engine.py
"""Handwritten Arabic pages: Kraken line segmentation + sherif v3 multi-pass + word vote.

Benchmark — TC01 (1972 handwritten land-sale contract, ballpoint on ruled paper,
phone photo; reference = human transcript, 295 words). Order-free word F1:

    pipeline                                              F1     body recall
    Qari v0.4, repo segmenter (previous default)         14.6%     18.3%
    KHATT CRNN, Kraken lines                             17.5%     19.0%
    Warraq-HTR-7B (4-bit), Kraken lines                  30.5%     34.2%
    sherif v3, Kraken lines                              47.7%     51.7%
    sherif v3, whole page (loops trimmed)                51.1%     55.1%
    sherif v3, 5 passes + word vote (this module)        56.7%     60.8%
      ... words left unflagged by the vote: ~72% correct; words flagged [؟]: ~14%

Findings that shaped the design:
  * The repo segmenter follows the RULINGS of lined paper and cuts handwriting in
    half; Kraken's neural baseline segmenter (blla) follows the slanted ink lines.
  * Handwriting-specific VLMs matter more than size: sherif v3 (Qwen2.5-VL-3B,
    trained on Muharaf + KHATT) beats the printed-text Qari and the 7B Warraq.
  * Ink "enhancement" (red channel, background flattening) and small-LLM
    post-correction (Qwen3-4B) both made things worse; the LLM also invented text.
  * Different views of the page (whole page, 4/6-line blocks, single lines at two
    heights) make partly independent mistakes, so a majority vote over words
    recovers ~9 F1 points, and disagreement is a calibrated "check this" signal.

Requires the [vlm] extra and, for segmentation, a Kraken virtualenv whose python
is given by ARABICOCR_KRAKEN_PY (falls back to the repo segmenter otherwise).
"""

import json
import os
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .vlm_engine import LazyVLM, nf4_config, prepare_line

SHERIF_ID = os.environ.get("ARABICOCR_HTR_MODEL", "sherif1313/Arabic-English-handwritten-OCR-v3")
SHERIF_PROMPT = ("ارجو استخراج النص العربي كاملاً من هذه الصورة من البداية الى النهاية بدون اي "
                 "اختصار ودون ذيادة او حذف. اقرأ كل المحتوى النصي الموجود في الصورة:")
KRAKEN_PY = os.environ.get("ARABICOCR_KRAKEN_PY")
_WORKER = Path(__file__).with_name("kraken_worker.py")

# (kind, arg): the first pass is the backbone — its line structure is the output's.
PASSES: Sequence[Tuple[str, float]] = (
    ("lines", 80), ("page", 1.0), ("block4", 1.5), ("lines", 112), ("block6", 1.5))
UNSURE = "[؟]"


# ---------------- segmentation ----------------

def kraken_lines(img: Image.Image) -> Optional[List[dict]]:
    """Kraken blla polygons, or None when no Kraken venv is configured / it fails."""
    if not KRAKEN_PY or not Path(KRAKEN_PY).exists():
        return None
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        img.convert("RGB").save(f.name)
        r = subprocess.run([KRAKEN_PY, str(_WORKER), f.name], capture_output=True, text=True,
                           timeout=600)
    if r.returncode != 0:
        return None
    return json.loads(r.stdout) or None


def poly_crop(img: Image.Image, boundary, pad: int = 6) -> Image.Image:
    """Line polygon -> its bounding-box crop, outside of the (slightly dilated)
    polygon painted paper-coloured so neighbouring lines' ascenders and
    descenders don't leak in."""
    W, H = img.size
    xs, ys = zip(*boundary)
    x0, y0 = max(0, min(xs) - pad), max(0, min(ys) - pad)
    x1, y1 = min(W, max(xs) + pad), min(H, max(ys) + pad)
    crop = img.crop((x0, y0, x1, y1)).convert("RGB")
    mask = Image.new("L", crop.size, 0)
    ImageDraw.Draw(mask).polygon([(x - x0, y - y0) for x, y in boundary], fill=255)
    mask = mask.filter(ImageFilter.MaxFilter(7))
    paper = tuple(int(v) for v in np.percentile(np.asarray(crop).reshape(-1, 3), 75, axis=0))
    return Image.composite(crop, Image.new("RGB", crop.size, paper), mask)


def line_blocks(img: Image.Image, lines: List[dict], n: int, scale: float) -> List[Image.Image]:
    """Unmasked union crop of every n consecutive lines, upscaled (context for the VLM)."""
    out = []
    for i in range(0, len(lines), n):
        xs, ys = zip(*[p for ln in lines[i:i + n] for p in ln["boundary"]])
        c = img.crop((max(0, min(xs) - 8), max(0, min(ys) - 8),
                      min(img.width, max(xs) + 8), min(img.height, max(ys) + 8))).convert("RGB")
        out.append(c.resize((round(c.width * scale), round(c.height * scale)), Image.LANCZOS))
    return out


# ---------------- recognizer ----------------

def trim_loops(text: str) -> str:
    """Collapse consecutive repeated lines (generation loops)."""
    out: List[str] = []
    for ln in text.split("\n"):
        if ln.strip() and out and ln.strip() == out[-1].strip():
            continue
        out.append(ln)
    return "\n".join(out)


class SherifHTR(LazyVLM):
    """sherif1313/Arabic-English-handwritten-OCR-v3 (Qwen2.5-VL-3B) in NF4, ~3.2 GB."""

    def _build(self):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            SHERIF_ID, torch_dtype=torch.float16, device_map={"": self.device},
            attn_implementation="sdpa", quantization_config=nf4_config()).eval()
        proc = AutoProcessor.from_pretrained(SHERIF_ID)
        proc.tokenizer.padding_side = "left"
        return model, proc

    def generate(self, imgs: List[Image.Image], max_new: int) -> List[str]:
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        class StopOnLoop(StoppingCriteria):
            """Stop once the last n tokens repeat twice more (n in 4..40) — whole-page
            passes otherwise loop on signatures until max_new_tokens."""

            def __init__(self, start):
                self.start = start

            def __call__(self, ids, scores, **kw):
                done = []
                for row in ids[:, self.start:].tolist():
                    done.append(any(len(row) >= 3 * n and row[-n:] == row[-2 * n:-n] == row[-3 * n:-2 * n]
                                    for n in range(4, 41)))
                return torch.tensor(done, device=ids.device)

        msgs = [[{"role": "user", "content": [{"type": "image", "image": im},
                                              {"type": "text", "text": SHERIF_PROMPT}]}] for im in imgs]
        prompts = [self.proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                   for m in msgs]
        inp = self.proc(text=prompts, images=imgs, padding=True, return_tensors="pt").to(self.model.device)
        n0 = inp["input_ids"].shape[1]
        with torch.inference_mode():
            out = self.model.generate(**inp, max_new_tokens=max_new, do_sample=False,
                                      repetition_penalty=1.1,
                                      stopping_criteria=StoppingCriteriaList([StopOnLoop(n0)]))
        return [trim_loops(t.strip()) for t in
                self.proc.batch_decode(out[:, n0:], skip_special_tokens=True)]


_sherif: Optional[SherifHTR] = None


def get_htr(device: str = "cuda") -> SherifHTR:
    global _sherif
    if _sherif is None:
        _sherif = SherifHTR(device)
    return _sherif


# ---------------- word vote ----------------

_DIAC = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")
_DIG = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def vote_key(w: str) -> str:
    """Spelling-tolerant key: digits, diacritics, hamza/alef/ya/ta-marbuta variants unified."""
    w = _DIAC.sub("", w.translate(_DIG))
    w = re.sub("[أإآٱ]", "ا", w).replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    return re.sub(r"[^\w]|_", "", w)


def vote_lines(backbone: List[str], others: List[str], min_votes: int = 2) -> Tuple[List[str], int]:
    """ROVER-style vote. Each other pass is word-aligned to the backbone (Levenshtein
    over normalized words); per backbone word the majority spelling wins, a word no
    other pass confirms is kept but tagged [؟]. Punctuation passes through."""
    from rapidfuzz.distance import Levenshtein

    toks = [(li, w) for li, ln in enumerate(backbone) for w in ln.split()]
    idx = [k for k, (_, w) in enumerate(toks) if vote_key(w)]
    keys = [vote_key(toks[k][1]) for k in idx]
    aligned = []
    for t in others:
        ow = [w for w in t.split() if vote_key(w)]
        ok = [vote_key(w) for w in ow]
        m = {}
        for tag, i1, i2, j1, j2 in Levenshtein.opcodes(keys, ok):
            if tag in ("equal", "replace") and i2 - i1 == j2 - j1:
                m.update({i1 + d: j1 + d for d in range(i2 - i1)})
        aligned.append((ow, ok, m))

    voted = {k: w for k, (_, w) in enumerate(toks)}
    flagged = 0
    for pos, k in enumerate(idx):
        votes, surface = Counter([keys[pos]]), {keys[pos]: toks[k][1]}
        for ow, ok, m in aligned:
            j = m.get(pos)
            if j is not None:
                votes[ok[j]] += 1
                surface.setdefault(ok[j], ow[j])
        best, n = votes.most_common(1)[0]
        if n >= min_votes:
            voted[k] = surface[best]
        else:
            voted[k] = f"{toks[k][1]} {UNSURE}"
            flagged += 1
    out = [[] for _ in backbone]
    for k, (li, _) in enumerate(toks):
        out[li].append(voted[k])
    return [" ".join(ws) for ws in out], flagged


# ---------------- pipeline ----------------

def recognize_handwritten(img: Image.Image, device: str = "cuda",
                          passes: Sequence[Tuple[str, float]] = PASSES,
                          progress: Optional[Callable[[float, str], None]] = None):
    """Returns (line_images, voted_line_texts, n_flagged_words, used_kraken)."""
    say = progress or (lambda f, msg: None)
    img = img.convert("RGB")
    say(0.0, "Segmenting lines (Kraken)…")
    polys = kraken_lines(img)
    if polys:
        lines = [poly_crop(img, p["boundary"]) for p in polys]
    else:  # no Kraken venv: degrade to the repo segmenter (weak on ruled paper)
        from .pipeline import segment_into_lines
        lines = segment_into_lines(img)
        polys = None

    eng = get_htr(device)
    with eng.lock:
        if not eng.loaded:
            say(0.05, "Loading sherif v3 handwriting model (first use, ~40 s)…")
        eng.ensure_loaded()
        texts = []
        for p, (kind, arg) in enumerate(passes):
            say(0.1 + 0.85 * p / len(passes), f"Pass {p + 1}/{len(passes)}: {kind} @{arg}")
            if kind == "lines":
                prepped = [prepare_line(ln, int(arg)) for ln in lines]
                outs = []
                for i in range(0, len(prepped), 4):
                    outs += eng.generate(prepped[i:i + 4], 160)
                texts.append(outs)
            elif kind == "page":
                s = float(arg)
                pg = img if s == 1 else img.resize((round(img.width * s), round(img.height * s)),
                                                   Image.LANCZOS)
                texts.append(eng.generate([pg], 1500)[0])
            elif kind.startswith("block") and polys:
                n = int(kind[5:])
                texts.append("\n".join(eng.generate([b], 120 * n)[0]
                                       for b in line_blocks(img, polys, n, float(arg))))
        eng.touch()

    backbone = texts[0] if isinstance(texts[0], list) else texts[0].split("\n")
    others = ["\n".join(t) if isinstance(t, list) else t for t in texts[1:]]
    voted, flagged = vote_lines(backbone, others)
    say(1.0, "Done")
    return lines, voted, flagged, polys is not None
