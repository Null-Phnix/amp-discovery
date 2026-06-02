"""Sanity check: verify environment, data, and GPU availability."""
import sys, platform
from pathlib import Path
print("python       {}".format(platform.python_version()))
print("platform     {}".format(platform.platform()))
try:
    import torch
    cuda_ok = torch.cuda.is_available()
    name = torch.cuda.get_device_name(0) if cuda_ok else "none"
    print("torch        {} cuda {} ({})".format(torch.__version__, cuda_ok, name))
except ImportError:
    print("torch        NOT INSTALLED (uv sync --extra train)")
data = Path(__file__).parent.parent / "data"
for f in ["Fold1.tab","Fold2.tab","Test.tab"]:
    p = data / f
    if p.exists():
        print("data/{}  {} KB".format(f, p.stat().st_size // 1024))
    else:
        print("data/{}  MISSING".format(f))
ready = (data / "Fold1.tab").exists()
print("AMP Discovery ready." if ready else "Run: ls data/")
