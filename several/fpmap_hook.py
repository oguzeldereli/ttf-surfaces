"""fpmap_hook.py (19 Nov 2026): snapshots for the combined fp -> rounding -> ternary map.
Wraps ttf.train; at every evaluation saves the model state to LS_SNAP/sNNN.pt with its validation loss (LS_ITERS
batches, fixed windows). In --mode fp32 it also measures the loss of the absmean rounding of the same weights (the
convert15.py rounding: T = clip(round(W / mean|W_i|)), W_r = alpha_i T with alpha_i the mean |W| over T != 0), for the
layers that are ternary in a TTF checkpoint of the same shape (LS_FMT).
Use: LS_SNAP=dir [LS_FMT=fmt.pt] python fpmap_hook.py <ttf.train arguments>"""
import os, sys, json
sys.path.insert(0, os.getcwd())
import torch
import ttf.train as TT

E = os.environ.get
SNAP, ITERS = E("LS_SNAP"), int(E("LS_ITERS", "4"))
os.makedirs(SNAP, exist_ok=True)
_orig = TT.evaluate
S = {"c": 0, "keys": None}


@torch.no_grad()
def hook(model, data, tc, device):
    c = S["c"]; S["c"] += 1
    it0 = tc.eval_iters; tc.eval_iters = ITERS
    loss = _orig(model, data, tc, device)
    rec = {"call": c, "loss": loss}
    sd = model.state_dict()
    if E("LS_FMT"):
        if S["keys"] is None:
            fm = torch.load(E("LS_FMT"), map_location="cpu", weights_only=False)["model"]
            want = {k[:-len(".wpacked")].split("_orig_mod.")[-1] for k in fm if k.endswith(".wpacked")}
            S["keys"] = [k for k in sd if k.endswith(".weight") and k[:-len(".weight")].split("_orig_mod.")[-1] in want]
            print(f"fpmap: {len(S['keys'])} layers rounded for the rounded loss", flush=True)
        params = dict(model.named_parameters())
        keep = {}
        for k in S["keys"]:
            W = params[k] if k in params else None
            if W is None:
                continue
            keep[k] = W.detach().clone()
            g = W.abs().mean(1, keepdim=True).clamp_min(1e-12)
            T = (W / g).round().clamp(-1, 1)
            nz = T != 0
            alpha = (W.abs() * nz).sum(1, keepdim=True) / nz.sum(1, keepdim=True).clamp_min(1)
            W.copy_(alpha * T)
        rec["rounded_loss"] = _orig(model, data, tc, device)
        for k, v in keep.items():
            params[k].copy_(v)
    torch.save({"rec": rec, "model": {k: v.detach().cpu().clone() for k, v in sd.items()}}, f"{SNAP}/s{c:03d}.pt")
    print("fpmap", json.dumps(rec), flush=True)
    tc.eval_iters = it0
    return loss


if E("LS_DSEED"):   # same init (torch.manual_seed(--seed)), different data order: remap only the batch sampler's seed
    _G, _ds = torch.Generator, int(E("LS_DSEED"))
    _tgt = int(sys.argv[sys.argv.index("--seed") + 1])

    class _Gen(_G):
        def manual_seed(self, s):
            return super().manual_seed(_ds if s == _tgt else s)
    torch.Generator = _Gen
    print(f"fpmap: data order seed {_ds} (init seed {_tgt})", flush=True)
TT.evaluate = hook
sys.argv = ["ttf.train"] + sys.argv[1:]
TT.main()
