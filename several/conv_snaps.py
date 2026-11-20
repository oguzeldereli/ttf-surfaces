"""conv_snaps.py: a directory of fp snapshots (fpmap_hook / fpsearch_hook: {"rec", "model"}) -> the same snapshots as
TTF checkpoints by the converter's absmean rounding (convert15.py), keeping "rec" (fp loss and rounded loss).
python conv_snaps.py FP_SNAP_DIR FMT_CKPT OUT_DIR"""
import glob, os, sys
import torch
from ttf.kernel import pack_rows

src, fmt, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
M = torch.load(fmt, map_location="cpu", weights_only=False)["model"]
st = lambda k: k.split("_orig_mod.")[-1]
for f in sorted(glob.glob(f"{src}/s*.pt")):
    d = torch.load(f, map_location="cpu", weights_only=False)
    F = {k.split("_orig_mod.")[-1]: v for k, v in d["model"].items()}
    o = {}
    for k, v in M.items():
        if k.endswith(".wpacked"):
            p = k[:-len(".wpacked")]
            W = F[st(p) + ".weight"].float()
            N = W.shape[0]
            g = W.abs().mean(1, keepdim=True).clamp_min(1e-12)
            T = (W / g).round().clamp(-1, 1)
            nz = T != 0
            alpha = (W.abs() * nz).sum(1) / nz.sum(1).clamp_min(1)
            beta = float((nz.sum().float() / N).clamp_min(1.0).rsqrt())
            o[k] = pack_rows(T.to(torch.int8))
            o[p + ".row_scale"] = alpha / beta
        elif k.endswith((".row_scale", ".col_scale", ".mag_A", ".mag_B")):
            p, s = k.rsplit(".", 1)
            if s == "row_scale":
                continue
            o[k] = torch.ones_like(v) if s == "col_scale" else (torch.zeros_like(v) if s == "mag_A" else v)
        elif st(k) in F and F[st(k)].shape == v.shape:
            o[k] = F[st(k)].to(v.dtype)
        else:
            o[k] = v
    torch.save({"rec": d["rec"], "model": o}, f"{out}/{os.path.basename(f)}")
print("converted", len(glob.glob(f"{out}/s*.pt")), "snapshots ->", out)
