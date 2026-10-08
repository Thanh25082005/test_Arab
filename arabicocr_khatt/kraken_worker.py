# arabicocr_khatt/kraken_worker.py
"""Kraken baseline (blla) line segmentation — run as a FILE inside the Kraken venv.

    $ARABICOCR_KRAKEN_PY arabicocr_khatt/kraken_worker.py page.png  > lines.json

Kraken pins its own torch, so it lives in a separate virtualenv and this script
must not import the arabicocr_khatt package. Output: a JSON list of
{"baseline": [[x, y], ...], "boundary": [[x, y], ...]} in reading order.
"""
import json
import sys

from kraken import blla
from PIL import Image

im = Image.open(sys.argv[1]).convert("RGB")
seg = blla.segment(im, text_direction="horizontal-rl")
json.dump([{"baseline": [[int(x), int(y)] for x, y in ln.baseline],
            "boundary": [[int(x), int(y)] for x, y in ln.boundary]} for ln in seg.lines],
          sys.stdout)
