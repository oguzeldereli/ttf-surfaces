"""mid_split.py: which part makes the networks between two evaluations better than both? For every third segment
(t, t+1) of the training path it measures: the reloaded evaluation networks (sanity), half of the segment's trit moves
with the floats of t, of t+1 and halfway, and the trits of one evaluation with the floats of the other.
Use: LS_OUT=ms.json LS_LAST=40 python mid_split.py <ttf.train arguments>
"""
import math, os, sys, time
sys.path.insert(0, os.getcwd())
import torch
import ttf.train as TT
from ttf.kernel import pack_rows, unpack_rows
from ttf.layers import KernelTernaryLinear

E = os.environ.get
OUT, TAG = E("LS_OUT"), E("LS_TAG", "run")
ITERS, N, LAST = int(E("LS_ITERS", "4")), int(E("LS_N", "41")), int(E("LS_LAST"))
MARGIN = float(E("LS_MARGIN", "0.12"))
ANGLE = E("LS_ANGLE", "time")      # index: angle 180°·i/n (storage order); time: by when the trit made its final move
TURNS = float(E("LS_TURNS", "1.5"))
RMAX, KMAX = float(E("LS_RMAX", "0.6")), int(E("LS_KMAX", "3"))     # arm candidates per cell
FINE = [0.2, 0.4, 0.6, 0.8]                                           # in-between path networks per segment
_orig = TT.evaluate
S = {"recs": [], "states": [], "calls": 0, "T0": None, "L": None, "prev": None}


def _get(L):
    return torch.cat([unpack_rows(m.wpacked, m.K).flatten() for m in L])


def _set(L, v):
    o = 0
    for m in L:
        n_, K = m.wpacked.shape[0], m.K
        m.wpacked.copy_(pack_rows(v[o:o + n_ * K].view(n_, K).to(torch.int8))); o += n_ * K


def _window(C, c, r):
    tot = int(C[-1]); r = min(r, tot)
    base = int(C[c - 1]) if c > 0 else 0
    left = min(r // 2, base); right = min(r - left, tot - base); left = r - right
    lo = int(torch.searchsorted(C, torch.tensor(base - left + 1, device=C.device, dtype=C.dtype))) if left > 0 else c
    hi = int(torch.searchsorted(C, torch.tensor(base + right, device=C.device, dtype=C.dtype))) + 1 if right > 0 else c
    return min(lo, c), max(hi, c)



@torch.no_grad()
def split(model, data, tc, device):
    L = S["L"]; Ts = S["Ts"]; out = []
    ns = len(S["states"]) - 1
    for t in list(range(0, ns, 3)) + [ns - 1]:
        sa = {k: v.to(device) for k, v in S["states"][t].items()}
        sb = {k: v.to(device) for k, v in S["states"][t + 1].items()}
        Ta, Tb = Ts[t].to(device), Ts[t + 1].to(device); dif = Tb - Ta
        g = torch.Generator(device="cpu").manual_seed(100 + t)
        u = torch.rand(2, Ta.numel(), generator=g).to(device)
        s1 = torch.where(dif.abs() == 2, u.min(0).values, u[0]); s2 = u.max(0).values
        def trits(fc):
            k = (fc >= s1).to(Ta.dtype) + ((dif.abs() == 2) & (fc >= s2)).to(Ta.dtype)
            return Ta + dif.sign() * k
        def floats(fc):
            return {kk: (sa[kk] * (1 - fc) + sb[kk] * fc).to(sa[kk].dtype) if sa[kk].is_floating_point() else sa[kk]
                    for kk in sa}
        r = {"t": t, "rec_a": S["recs"][t]["loss"], "rec_b": S["recs"][t + 1]["loss"]}
        for name, tf, ff in [("a_reload", 0.0, 0.0), ("b_reload", 1.0, 1.0), ("mid", 0.5, 0.5), ("tmid_fa", 0.5, 0.0),
                             ("ta_fmid", 0.0, 0.5), ("tmid_fb", 0.5, 1.0), ("tb_fa", 1.0, 0.0), ("ta_fb", 0.0, 1.0)]:
            model.load_state_dict(floats(ff)); _set(L, trits(tf))
            r[name] = _orig(model, data, tc, device)
        out.append(r)
        print("mid_split", TAG, " ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in r.items()), flush=True)
    import json
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"mid_split {TAG}: wrote {OUT}", flush=True)


@torch.no_grad()
def hook(model, data, tc, device):
    c = S["calls"]; S["calls"] += 1
    if S["L"] is None:
        S["L"] = [m for m in model.modules() if isinstance(m, KernelTernaryLinear)]; S["Ts"] = []
    it0 = tc.eval_iters; tc.eval_iters = ITERS
    loss0 = _orig(model, data, tc, device)
    S["recs"].append({"loss": loss0}); S["Ts"].append(_get(S["L"]).cpu())
    S["states"].append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    print(f"mid_split {TAG}: call {c} loss {loss0:.4f}", flush=True)
    if c == LAST - 1:
        cur = S["states"][-1]
        split(model, data, tc, device)
        model.load_state_dict({k: v.to(device) for k, v in cur.items()})
    tc.eval_iters = it0
    return loss0


TT.evaluate = hook
sys.argv = ["ttf.train"] + sys.argv[1:]
TT.main()
