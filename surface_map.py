"""surface_map.py: the measured loss surface around a ternary training path (Guzeldereli 2026, paper 3).

Wraps the TTF trainer (ttf.train). At every evaluation it records the loss and a snapshot of the model state; at the
LS_LAST-th evaluation it builds the surface and saves it to LS_OUT, then training continues.

Projection: every trit has one fixed unit vector in the plane; a +1 move is one step along it, a -1 move one step the
opposite way, and a network's position is the sum of its moves from the start network (LS_T0, else the first
evaluation). With LS_ANGLE=time (default) a trit's angle is set by when it made its final move (LS_TURNS turns over the
run) and points along its net move, so the moves of one period add up; trits that end where they started get random
angles. LS_ANGLE=index gives angle 180 deg * i / n (storage order) for comparison.

Surface: for every grid point (LS_N x LS_N over the path's extent + LS_MARGIN) and every nearby arm of the path (local
minima over path segments of the distance, up to LS_KMAX, within LS_RMAX of the grid half-width), the real network
part of the way between the two evaluations of that segment (each differing trit moves at its own random time; floats
interpolated), then moved sideways to the grid point with the fewest legal single moves (those whose directions are
closest to the needed direction). Four in-between networks per segment (no sideways moves) trace the path itself.
Every height is a measured loss on LS_ITERS fixed validation batches.

Output (torch.save): cell (grid index, -1 for path networks), seg, frac, loss, qx, qy (position), moves (sideways
moves), recs (per evaluation: loss, pos, moves_from_start, moves_since_prev), N, x, y.
Use: LS_OUT=sf.pt LS_LAST=40 python surface_map.py <ttf.train arguments>   (run from the trainer's directory)
For models of ~1e9 trits use surface_map_big.py.
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
def grid(model, data, tc, device):
    t0 = time.time()
    L = S["L"]
    n = S["T0"].numel()
    dev = S["T0"].device
    T0 = S["T0"]
    Ts = []
    for st in S["states"]:                                  # trit vectors kept on the CPU (large models)
        model.load_state_dict({k: v.to(device) for k, v in st.items()})
        Ts.append(_get(L).cpu())
    torch.cuda.empty_cache()
    if ANGLE == "time":
        # each trit's vector: angle by WHEN it made its final move (rotating with training time, TURNS turns over the
        # run), pointing along its net move (so a trit's net move adds +1 unit toward its angle); trits that end where
        # they started get random angles. Still one fixed unit vector per trit, +1 one way, -1 the opposite way.
        tl = torch.full((n,), -1, device=dev, dtype=torch.int32)
        prev = T0
        for t, T in enumerate(Ts):
            T = T.to(dev)
            tl = torch.where(T != prev, torch.full_like(tl, t), tl); prev = T
        del prev
        net = Ts[-1].to(dev) - T0
        sg = torch.where(net < 0, -1.0, 1.0).double()
        mov = (tl >= 0) & (net != 0)
        key = torch.where(mov, tl.long() * n + torch.arange(n, device=dev), torch.full((n,), 2 ** 62, device=dev))
        rank = torch.empty(n, device=dev, dtype=torch.float64)
        rank[key.argsort()] = torch.arange(n, device=dev, dtype=torch.float64)
        nm = int(mov.sum())
        g = torch.Generator(device="cpu").manual_seed(5)
        phi = torch.where(mov, 2 * math.pi * TURNS * rank / max(nm, 1),
                          (torch.rand(n, generator=g, dtype=torch.float64) * 2 * math.pi).to(dev))
        del rank, key, tl, net
        cs, sn = (sg * phi.cos()).float(), (sg * phi.sin()).float()
        del phi, sg, mov
        print(f"surface_map {TAG}: time angles, {nm} trits with a net move, {TURNS} turns", flush=True)
    else:
        phi = torch.arange(n, device=dev, dtype=torch.float64) * (math.pi / n)
        cs, sn = phi.cos(), phi.sin()
    for t, T in enumerate(Ts):
        Dv = (T.to(dev) - T0).double()
        S["recs"][t]["pos"] = (float((Dv * cs).sum()), float((Dv * sn).sum()))
    del Dv
    # the 2n single moves sorted by the direction they point (for the sideways shifts)
    al = torch.atan2(sn, cs) % (2 * math.pi)
    dirs = torch.cat([al, (al + math.pi) % (2 * math.pi)]); del al
    order = dirs.argsort()
    dsorted = dirs[order]; del dirs
    pdel = torch.where(order < n, 1, -1).to(Ts[0].dtype)
    pidx = (order % n).int(); del order
    torch.cuda.empty_cache()
    P = torch.tensor([r["pos"] for r in S["recs"]], dtype=torch.float64)
    lo, hi = P.min(0).values, P.max(0).values
    ctr, half = (lo + hi) / 2, float((hi - lo).max()) / 2 * (1 + 2 * MARGIN)
    xs = torch.linspace(float(ctr[0]) - half, float(ctr[0]) + half, N)
    ys = torch.linspace(float(ctr[1]) - half, float(ctr[1]) + half, N)
    Q = torch.stack(torch.meshgrid(xs, ys, indexing="xy"), -1).view(-1, 2).double()
    # every cell gets one network per nearby ARM of the path: each local minimum over segments (t, t+1) of the distance
    # to the path (within RMAX of the grid half-width, at most KMAX per cell). Where arms lie side by side, a cell is
    # built from each, so every sheet continues past the boundary and its rise away from its own arm is measured
    # (layers), instead of stopping where another arm becomes nearer.
    A, B = P[:-1], P[1:]
    AB = B - A
    f = (((Q[:, None] - A[None]) * AB[None]).sum(-1) / (AB * AB).sum(-1).clamp_min(1e-9)[None]).clamp(0, 1)
    dist = (Q[:, None] - (A[None] + f[..., None] * AB[None])).norm(dim=-1)          # [cells, segments]
    ns = dist.shape[1]
    pad = torch.full((dist.shape[0], 1), float("inf"), dtype=dist.dtype)
    dl, dr = torch.cat([pad, dist[:, :-1]], 1), torch.cat([dist[:, 1:], pad], 1)
    locmin = (dist <= dl) & (dist <= dr) & (dist <= RMAX * half)
    locmin[torch.arange(len(Q)), dist.argmin(1)] = True
    cand = []                                               # (cell, segment, frac)
    for ci in range(len(Q)):
        ts = locmin[ci].nonzero().flatten()
        kept = []
        for t in ts[dist[ci, ts].argsort()].tolist():      # nearest first; a second candidate only from another arm
            if all(abs(t - k) > 2 for k in kept):
                kept.append(t)
            if len(kept) == KMAX:
                break
        cand += [(ci, t, float(f[ci, t])) for t in kept]
    # finer path: in-between networks at fractions FINE of every segment (no sideways moves)
    for t in range(ns):
        cand += [(-1, t, fr) for fr in FINE]
    nc = len(cand)
    c_cell = torch.tensor([c[0] for c in cand]); c_seg = torch.tensor([c[1] for c in cand])
    c_frac = torch.tensor([c[2] for c in cand]); c_loss = torch.full((nc,), float("nan"))
    c_x = torch.zeros(nc); c_y = torch.zeros(nc); c_mv = torch.zeros(nc)
    print(f"surface_map {TAG}: {nc} networks to build ({(c_cell >= 0).sum()} surface, {(c_cell < 0).sum()} path)",
          flush=True)
    for t in range(ns):
        idx = (c_seg == t).nonzero().flatten().tolist()
        if not idx:
            continue
        sa = {k: v.to(device) for k, v in S["states"][t].items()}
        sb = {k: v.to(device) for k, v in S["states"][t + 1].items()}
        Ta, Tb = Ts[t].to(dev), Ts[t + 1].to(dev)
        dif = Tb - Ta
        g = torch.Generator(device="cpu").manual_seed(100 + t)
        u = torch.rand(2, n, generator=g).to(dev)
        s1 = torch.where(dif.abs() == 2, u.min(0).values, u[0]); s2 = u.max(0).values
        for j in idx:
            fc = float(c_frac[j]); ci = int(c_cell[j])
            # base: the real network fc of the way from path network t to t+1 (each differing trit moves at its own
            # random time, so fc of the actual moves are made); float parts interpolated the same way
            k = (fc >= s1).to(Ta.dtype) + ((dif.abs() == 2) & (fc >= s2)).to(Ta.dtype)
            v = Ta + dif.sign() * k
            model.load_state_dict({kk: (sa[kk] * (1 - fc) + sb[kk] * fc).to(sa[kk].dtype)
                                   if sa[kk].is_floating_point() else sa[kk] for kk in sa})
            r = 0
            if ci >= 0:
                Dv = (v - S["T0"]).double()
                pb = torch.stack([(Dv * cs).sum(), (Dv * sn).sum()]).cpu()
                d = Q[ci] - pb
                r = int(round(float(d.norm())))
                if r > 0:  # then sideways to the grid point with the fewest moves (the legal single moves pointing
                    psi = math.atan2(float(d[1]), float(d[0])) % (2 * math.pi)   # closest to that direction)
                    c = min(int(torch.searchsorted(dsorted, torch.tensor(psi, device=dev, dtype=dsorted.dtype))), 2 * n - 1)
                    nv = v[pidx] + pdel
                    lg = (nv >= -1) & (nv <= 1); del nv
                    C = lg.int().cumsum(0, dtype=torch.int32)
                    lo_, hi_ = _window(C, c, r)
                    sel = torch.arange(lo_, hi_, device=dev)[lg[lo_:hi_]]
                    v[pidx[sel].long()] += pdel[sel]
            _set(L, v)
            c_loss[j] = _orig(model, data, tc, device)
            Dv = (v - S["T0"]).double()
            c_x[j], c_y[j] = float((Dv * cs).sum()), float((Dv * sn).sum())
            c_mv[j] = r
        print(f"surface_map {TAG}: segment {t}: {len(idx)} networks, {time.time() - t0:.0f}s", flush=True)
    torch.save({"tag": TAG, "n": n, "N": N, "x": xs, "y": ys, "cell": c_cell, "seg": c_seg, "frac": c_frac,
                "loss": c_loss, "qx": c_x, "qy": c_y, "moves": c_mv, "recs": S["recs"], "version": 4}, OUT)
    print(f"surface_map {TAG}: wrote {OUT} ({nc} networks, {time.time() - t0:.0f}s)", flush=True)


@torch.no_grad()
def hook(model, data, tc, device):
    c = S["calls"]; S["calls"] += 1
    if S["L"] is None:
        S["L"] = [m for m in model.modules() if isinstance(m, KernelTernaryLinear)]
    L = S["L"]
    it0 = tc.eval_iters; tc.eval_iters = ITERS
    T = _get(L)
    dev = T.device
    if S["T0"] is None:
        S["T0"] = T.clone()
        if E("LS_T0"):
            st = torch.load(E("LS_T0"), map_location="cpu", weights_only=False)["model"]
            parts = []
            for name, m in model.named_modules():
                if isinstance(m, KernelTernaryLinear):
                    key = next(k for k in st if k.endswith(name.split("_orig_mod.")[-1] + ".wpacked"))
                    parts.append(unpack_rows(st[key].to(dev), m.K).flatten())
            S["T0"] = torch.cat(parts)
    n = T.numel()
    phi = torch.arange(n, device=dev, dtype=torch.float64) * (math.pi / n)
    D = (T - S["T0"]).double()
    loss0 = _orig(model, data, tc, device)
    rec = {"call": c, "loss": loss0, "pos": (float((D * phi.cos()).sum()), float((D * phi.sin()).sum())),
           "moves_from_start": float(D.abs().sum()),
           "moves_since_prev": float((T - S["prev"]).abs().sum()) if S["prev"] is not None else 0.0}
    S["prev"] = T.clone()
    S["recs"].append(rec)
    S["states"].append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    print(f"surface_map {TAG}: call {c} loss {loss0:.4f} pos ({rec['pos'][0]:.0f}, {rec['pos'][1]:.0f})", flush=True)
    if c == LAST - 1:
        cur = S["states"][-1]
        grid(model, data, tc, device)
        model.load_state_dict({k: v.to(device) for k, v in cur.items()})
    tc.eval_iters = it0
    return loss0


TT.evaluate = hook
sys.argv = ["ttf.train"] + sys.argv[1:]
TT.main()
