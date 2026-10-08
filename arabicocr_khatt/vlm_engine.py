# arabicocr_khatt/vlm_engine.py
"""Line-level Arabic OCR with a vision-language model (Qari-OCR v0.4).

The KHATT CRNN is a handwriting model; on typed / printed documents (1970s
typewriter pages, phone photos) it misses most words. Benchmarked on two such
pages (39 lines, 501 words, hand-made ground truth):

    engine                         CER(n)   words recovered
    KHATT CRNN                     52.4%     9.4%
    Tesseract (ara, per line)      15.4%    52.6%
    PaddleOCR-VL 0.9B (per line)   12.1%    60.7%
    Qwen3-VL-4B base (per line)    18.4%    58.4%
    Qari-OCR v0.4 4-bit (per line) 10.5%    71.8%   <- this module

Reading line by line (repo segmenter, each line scaled to ~80 px) beats whole-page
prompts: every line is read exactly once (no skipped or looped lines) and memory
stays bounded. Taller lines (112/128 px), 64 px lines, contrast enhancement and
3-line chunks all scored lower.

Requires: transformers>=4.57 (Qwen3-VL), bitsandbytes, peft. The model is loaded
lazily (4-bit NF4, ~3.6 GB VRAM) and released after ARABICOCR_VLM_IDLE_S seconds
without use, so it does not hold GPU memory on a shared machine.
"""

import os
import threading
import time
from typing import Callable, List, Optional

from PIL import Image

BASE_ID = os.environ.get("ARABICOCR_VLM_BASE", "Qwen/Qwen3-VL-4B-Instruct")
ADAPTER_ID = os.environ.get("ARABICOCR_VLM_ADAPTER", "YasserSami/Qari-OCR-0.4.0-VL-4B-Instruct")
PROMPT = "Free OCR."           # the adapter's training prompt
LINE_H = 80                    # px; best of 64/80/112/128 in the benchmark above
MAX_NEW_TOKENS = 200           # per line; bounds runaway repetition
BATCH = int(os.environ.get("ARABICOCR_VLM_BATCH", "4"))
BUDGET_GB = float(os.environ.get("ARABICOCR_VLM_BUDGET_GB", "4.5"))
IDLE_S = float(os.environ.get("ARABICOCR_VLM_IDLE_S", "600"))


def prepare_line(ln: Image.Image, line_h: int = LINE_H) -> Image.Image:
    """Scale a line crop to ~line_h px tall and add a white margin."""
    ln = ln.convert("RGB")
    s = max(1.0, line_h / max(1, ln.height))
    if s > 1.0:
        ln = ln.resize((round(ln.width * s), round(ln.height * s)), Image.LANCZOS)
    m = max(8, ln.height // 4)
    canvas = Image.new("RGB", (ln.width + 2 * m, ln.height + 2 * m), (255, 255, 255))
    canvas.paste(ln, (m, m))
    return canvas


class LazyVLM:
    """Lazy, thread-safe, auto-unloading holder for one 4-bit VLM.

    Subclasses implement ``_build() -> (model, processor)``. Only one LazyVLM is
    resident at a time (loading one unloads the others): two ~4 GB models do not
    fit next to each other on a shared T4.
    """

    _instances: List["LazyVLM"] = []

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self.proc = None
        self.lock = threading.RLock()
        self._last_used = 0.0
        self._timer: Optional[threading.Timer] = None
        LazyVLM._instances.append(self)

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def _build(self):
        raise NotImplementedError

    def ensure_loaded(self):
        if self.model is not None:
            return
        for other in LazyVLM._instances:
            if other is not self:
                other.unload()
        import torch
        dev = torch.device(self.device)
        if dev.type == "cuda":
            idx = dev.index if dev.index is not None else torch.cuda.current_device()
            total = torch.cuda.get_device_properties(idx).total_memory
            # Hard cap: on a shared GPU we should OOM ourselves, not the neighbour.
            torch.cuda.set_per_process_memory_fraction(min(1.0, BUDGET_GB * 2**30 / total), idx)
        self.model, self.proc = self._build()

    def unload(self):
        with self.lock:
            if self.model is None:
                return
            import gc
            import torch
            self.model = self.proc = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def touch(self):
        """Mark as used now and (re)arm the idle-unload timer."""
        self._last_used = time.time()
        if self._timer:
            self._timer.cancel()
        if IDLE_S > 0:
            self._timer = threading.Timer(IDLE_S, self._idle_check)
            self._timer.daemon = True
            self._timer.start()

    def _idle_check(self):
        with self.lock:
            if time.time() - self._last_used >= IDLE_S:
                self.unload()


def nf4_config(skip=("visual", "lm_head")):
    import torch
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16, llm_int8_skip_modules=list(skip))


class QariLineOCR(LazyVLM):
    """Qari-OCR v0.4 (Qwen3-VL-4B + Arabic LoRA), one line crop per request."""

    def _build(self):
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            BASE_ID, torch_dtype=torch.float16, device_map={"": self.device},
            attn_implementation="sdpa", quantization_config=nf4_config()).eval()
        model = PeftModel.from_pretrained(model, ADAPTER_ID).eval()
        proc = AutoProcessor.from_pretrained(ADAPTER_ID)
        proc.tokenizer.padding_side = "left"  # batched generation
        return model, proc

    def recognize_lines(self, lines: List[Image.Image],
                        progress: Optional[Callable[[int, int], None]] = None) -> List[str]:
        """OCR each line crop (one string per line, same order)."""
        import torch
        if not lines:
            return []
        with self.lock:
            self.ensure_loaded()
            prepped = [prepare_line(ln) for ln in lines]
            texts: List[str] = []
            for i in range(0, len(prepped), BATCH):
                chunk = prepped[i:i + BATCH]
                try:
                    texts += self._generate(chunk, torch)
                except torch.OutOfMemoryError:
                    # Shared GPU got tight: retry this batch one line at a time.
                    torch.cuda.empty_cache()
                    texts += [self._generate([im], torch)[0] for im in chunk]
                if progress:
                    progress(min(i + BATCH, len(prepped)), len(prepped))
            self.touch()
            return texts

    def _generate(self, imgs: List[Image.Image], torch) -> List[str]:
        msgs = [[{"role": "user", "content": [{"type": "image", "image": im},
                                              {"type": "text", "text": PROMPT}]}] for im in imgs]
        prompts = [self.proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                   for m in msgs]
        inputs = self.proc(text=prompts, images=imgs, padding=True,
                           return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                      do_sample=False, repetition_penalty=1.05)
        out = out[:, inputs["input_ids"].shape[1]:]
        return [t.strip() for t in self.proc.batch_decode(out, skip_special_tokens=True)]


_engine: Optional[QariLineOCR] = None


def get_engine(device: str = "cuda") -> QariLineOCR:
    global _engine
    if _engine is None:
        _engine = QariLineOCR(device)
    return _engine
