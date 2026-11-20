"""fpsearch_hook.py (19 Nov 2026): the 15M version of the fp searches for a network that rounds better, with snapshots for the combined map.
Starts from an fp32 checkpoint A (FS_INIT) and trains every weight W (AdamW, FS_LR, warmup 20, clip 1):
  twogoal  CE(W) + FS_LAM * KL(A's fp || rounded(W))      (rounding straight-through: W + (round(W) - W).detach())
  noise    CE(W + Z), Z fresh Gaussian of relative norm FS_NOISE per row of every ternary-to-be linear
  ste      CE(rounded(W))                                    (quantisation-aware training from the fp network)
round = the converter's absmean rounding (convert15.py): alpha_i * clip(round(W / mean|W_i|)).
Every FS_EVERY steps: snapshot (FS_SNAP/sNNN.pt: state + fp loss + rounded loss on ITERS fixed val batches).
Run as a ttf.train wrapper with the fp32 shape's arguments; all work happens at the first evaluation, then it exits."""
import os, sys, json, time
sys.path.insert(0, os.getcwd())
import numpy as np
import torch
from torch.func import functional_call
import ttf.train as TT

E = os.environ.get
MODE, INIT, SNAP, FMT = E("FS_MODE"), E("FS_INIT"), E("FS_SNAP"), E("FS_FMT")
STEPS, EVERY, ITERS = int(E("FS_STEPS", "1000")), int(E("FS_EVERY", "50")), int(E("FS_ITERS", "4"))
LR, LAM, NOISE = float(E("FS_LR", "1e-4")), float(E("FS_LAM", "1")), float(E("FS_NOISE", "0.3"))
MICRO, ACC, SEED = int(E("FS_MICRO", "2")), int(E("FS_ACC", "8")), int(E("FS_SEED", "5"))
os.makedirs(SNAP, exist_ok=True)
_orig = TT.evaluate
strip = lambda k: k.split("_orig_mod.")[-1]


def rnd(W):
    g = W.abs().mean(1, keepdim=True).clamp_min(1e-12)
    T = (W / g).round().clamp(-1, 1)
    nz = T != 0
    al = (W.abs() * nz).sum(1, keepdim=True) / nz.sum(1, keepdim=True).clamp_min(1)
    return al * T


def run(model, data, tc, device):
    t0 = time.time()
    m = getattr(model, "_orig_mod", model)
    tc.eval_iters = ITERS
    st = torch.load(INIT, map_location="cpu", weights_only=False)["model"]
    own = {strip(k): k for k in m.state_dict()}
    m.load_state_dict({own[strip(k)]: v for k, v in st.items() if strip(k) in own})
    fm = torch.load(FMT, map_location="cpu", weights_only=False)["model"]
    want = {strip(k[:-len(".wpacked")]) for k in fm if k.endswith(".wpacked")}
    P = dict(m.named_parameters())
    WK = [k for k in P if k.endswith(".weight") and strip(k[:-len(".weight")]) in want]
    print(f"fpsearch {MODE}: {len(WK)} linears rounded, lr {LR}, lam {LAM}, noise {NOISE}", flush=True)
    A = {k: v.detach().clone() for k, v in P.items()}
    opt = torch.optim.AdamW(list(P.values()), lr=LR, weight_decay=0.0, betas=(0.9, 0.95))
    tr = np.memmap(tc.data_path, dtype=np.uint16, mode="r")
    rng = np.random.default_rng(SEED)
    S = tc.seq_len

    @torch.no_grad()
    def snap(step):
        m.eval()
        l = _orig(model, data, tc, device)
        keep = {k: P[k].detach().clone() for k in WK}
        for k in WK:
            P[k].copy_(rnd(P[k]))
        lr_ = _orig(model, data, tc, device)
        for k in WK:
            P[k].copy_(keep[k])
        d = sum(float((P[k] - A[k]).norm() ** 2) for k in WK) ** 0.5 / sum(float(A[k].norm() ** 2) for k in WK) ** 0.5
        rec = {"call": step // EVERY, "step": step, "loss": l, "rounded_loss": lr_, "rel_dist": d}
        torch.save({"rec": rec, "model": {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}},
                   f"{SNAP}/s{step // EVERY:03d}.pt")
        print("fpsearch", json.dumps(rec), f"{time.time() - t0:.0f}s", flush=True)
        m.train()

    snap(0)
    m.train()
    with torch.enable_grad():
        for step in range(1, STEPS + 1):
            for g in opt.param_groups:
                g["lr"] = LR * min(1.0, step / 20)
            for _ in range(ACC):
                ix = rng.integers(0, len(tr) - S - 2, MICRO)
                x = torch.stack([torch.from_numpy(tr[i:i + S + 1].astype(np.int64)) for i in ix]).to(device)
                inp, tgt = x[:, :-1], x[:, 1:]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if MODE in ("twogoal", "ste"):
                        Wr = dict(P)
                        for k in WK:
                            Wr[k] = P[k] + (rnd(P[k].detach()) - P[k]).detach()
                    if MODE == "twogoal":
                        with torch.no_grad():
                            z, _ = functional_call(m, A, (inp,))
                            tlp = torch.log_softmax(z.float(), -1).reshape(-1, z.shape[-1]); del z
                        m.kd_logp, m.kd_w = tlp, 1.0
                        _, kl = functional_call(m, Wr, (inp, tgt))
                        m.kd_logp = None
                        _, ce = m(inp, tgt)
                        lo = LAM * kl + ce
                    elif MODE == "noise":
                        Wn = dict(P)
                        for k in WK:
                            with torch.no_grad():
                                Z = torch.randn_like(P[k])
                                Z = Z * (NOISE * P[k].norm(dim=1, keepdim=True) / Z.norm(dim=1, keepdim=True))
                            Wn[k] = P[k] + Z
                        _, lo = functional_call(m, Wn, (inp, tgt))
                    else:
                        _, lo = functional_call(m, Wr, (inp, tgt))
                (lo / ACC).backward()
            torch.nn.utils.clip_grad_norm_(list(P.values()), 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
            if step % EVERY == 0:
                snap(step)
    torch.save({"model": {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}}, f"{SNAP}/../end.pt")
    print(f"fpsearch {MODE}: done {time.time() - t0:.0f}s", flush=True)
    os._exit(0)


TT.evaluate = lambda model, data, tc, device: run(model, data, tc, device)
sys.argv = ["ttf.train"] + sys.argv[1:]
TT.main()
