"""convert15.py (18 Nov 2026): the 15M conversion testbed's converter.

A full-precision core checkpoint (--mode fp32, preset s15) -> a TTF checkpoint for --init-ckpt, by plain absmean rounding
(no rotations, no GPTQ: the testbed isolates the theta question):
  gamma_i = mean |W_i|, T = clip(round(W / gamma_i), -1, 1), alpha_i = mean |W_i| over T != 0 (least squares for fixed T),
  row_scale_i = alpha_i / beta (the layer's beta = 1 / sqrt(nnz / N) multiplies the trit product), col_scale = 1,
  float adapter mag_A = 0 (mag_B keeps the init's random values, as a fresh run: with both 0 it never trains).
Load with --resume (copy to OUT_DIR/ckpt.pt): --init-ckpt only loads --arch-hf models (18 Nov: native presets ignore it silently). Every other tensor (embedding, head, norms, qk scales) is copied.

python convert15.py FP_CKPT FMT_CKPT OUT_CKPT   (FMT: any TTF checkpoint of the same shape, e.g. --stop-after 0)
"""
import sys
import torch
from ttf.kernel import pack_rows, unpack_rows

fp = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
fm = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
F, M = fp["model"], fm["model"]
out, rel = {}, []
for k, v in M.items():
    if k.endswith(".wpacked"):
        p = k[:-len(".wpacked")]
        W = F[p + ".weight"].float()
        N, K = W.shape
        g = W.abs().mean(1, keepdim=True).clamp_min(1e-12)
        T = (W / g).round().clamp(-1, 1)
        nz = T != 0
        alpha = (W.abs() * nz).sum(1) / nz.sum(1).clamp_min(1)
        beta = float((nz.sum().float() / N).clamp_min(1.0).rsqrt())
        out[k] = pack_rows(T.to(torch.int8))
        assert torch.equal(unpack_rows(out[k], K).float(), T)
        out[p + ".row_scale"] = alpha / beta
        rel.append(float(((alpha[:, None] * T - W).norm() / W.norm()) ** 2))
    elif k.endswith((".row_scale", ".col_scale", ".mag_A", ".mag_B")):
        p, s = k.rsplit(".", 1)
        if s == "row_scale":
            continue
        out[k] = torch.ones_like(v) if s == "col_scale" else (torch.zeros_like(v) if s == "mag_A" else v)   # B: init's random
    elif k in F and F[k].shape == v.shape:
        out[k] = F[k].to(v.dtype)
    else:
        out[k] = v
        print("kept template:", k, tuple(v.shape))
miss = [k for k in F if k not in out and not k.endswith(".weight")]
print("fp tensors not used:", miss[:10])
fm["model"] = out
fm["step"] = 0
torch.save(fm, sys.argv[3])
print(f"converted {len(rel)} ternary layers; mean relative error^2 {sum(rel) / len(rel):.4f}")
