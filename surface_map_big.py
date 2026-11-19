"""surface_map_big.py: surface_map.py for large models (e.g. a 0.8B conversion recovery).
Same projection and surface as surface_map.py: one fixed unit vector per trit, angle by the time of its final move,
pointing along its net move; every surface network = the real in-between network on the nearest arm(s) of the path,
moved sideways to the grid point with the fewest legal single moves. Memory changes for ~1e9 trits:
  - snapshots keep only the packed trits and the trainable floats (the rest from the first snapshot), on the CPU;
  - trit vectors are unpacked on demand; time angles and unit vectors live on the CPU (float32);
  - a network's position is updated move by move (base = evaluation position + the moves made), never a full dot;
  - the sideways moves come from angle bins (NB bins, counting sort per sign) instead of one global sort.
Env: LS_OUT, LS_TAG, LS_LAST (evaluations; the grid is built at the last), LS_N (31), LS_ITERS (eval batches, 8),
LS_MARGIN, LS_TURNS, LS_RMAX, LS_KMAX, LS_T0 (checkpoint with the start trits; else the first evaluation), LS_NB."""
import math, os, sys, time
sys.path.insert(0, os.getcwd())
import numpy as np
import torch
import ttf.train as TT
from ttf.kernel import pack_rows, unpack_rows
from ttf.layers import KernelTernaryLinear

E = os.environ.get
OUT, TAG = E("LS_OUT"), E("LS_TAG", "run")
ITERS, N, LAST = int(E("LS_ITERS", "8")), int(E("LS_N", "31")), int(E("LS_LAST"))
MARGIN, TURNS = float(E("LS_MARGIN", "0.12")), float(E("LS_TURNS", "1.5"))
RMAX, KMAX, NB = float(E("LS_RMAX", "0.6")), int(E("LS_KMAX", "3")), int(E("LS_NB", "65536"))
FINE = [0.2, 0.4, 0.6, 0.8]
PROF = bool(int(E("LS_PROF", "0")))
_orig = TT.evaluate
S = {"recs": [], "snaps": [], "calls": 0, "T0": None, "L": None, "keys": None, "base": None}


def _get(L):
    return torch.cat([unpack_rows(m.wpacked, m.K).flatten() for m in L])


def _set(L, v):
    o = 0
    for m in L:
        n_, K = m.wpacked.shape[0], m.K
        m.wpacked.copy_(pack_rows(v[o:o + n_ * K].view(n_, K).to(torch.int8))); o += n_ * K


def _trits(t, dev):
    """trit vector of snapshot t (int8, on dev) from its packed tensors"""
    L, snap = S["L"], S["snaps"][t]
    return torch.cat([unpack_rows(snap[k].to(dev), m.K).flatten() for k, m in zip(S["pk"], L)])


def _floats(t, u=None, f=0.0):
    a = S["snaps"][t]
    if u is None:
        return {k: a[k] for k in S["fk"]}
    b = S["snaps"][u]
    return {k: (a[k].float() * (1 - f) + b[k].float() * f).to(a[k].dtype) for k in S["fk"]}


@torch.no_grad()
def grid(model, data, tc, device):
    t0 = time.time()
    L, dev = S["L"], device
    T0 = S["T0"]                                       # CPU int8
    n = T0.numel(); ne = len(S["snaps"])
    # time of each trit's final move, net move (CPU)
    tl = torch.full((n,), -1, dtype=torch.int16)
    prev = T0
    for t in range(ne):
        T = _trits(t, dev).cpu()
        tl[T != prev] = t; prev = T
    net = (prev.to(torch.int16) - T0.to(torch.int16))
    mov = (tl >= 0) & (net != 0)
    nm = int(mov.sum())
    # rank movers by (time, index): counting by time keeps it O(n)
    rank = torch.zeros(n, dtype=torch.float32)
    off = 0
    for t in range(ne):
        idx = (mov & (tl == t)).nonzero().flatten()
        rank[idx] = torch.arange(off, off + len(idx), dtype=torch.float32); off += len(idx)
    g = torch.Generator().manual_seed(5)
    phi = torch.rand(n, generator=g) * (2 * math.pi)
    phi[mov] = 2 * math.pi * TURNS * rank[mov] / max(nm, 1)
    sg = torch.where(net < 0, -1.0, 1.0)
    del rank, tl
    cs, sn = sg * phi.cos(), sg * phi.sin()
    print(f"surface_map_big {TAG}: time angles, {nm} trits with a net move of {n}, {time.time() - t0:.0f}s", flush=True)
    # positions of the evaluation networks (chunked dot)
    Ts_pos = []
    for t in range(ne):
        D = (_trits(t, dev).cpu().to(torch.int16) - T0.to(torch.int16))
        nz = D.nonzero().flatten(); dv = D[nz].double()
        Ts_pos.append((float((dv * cs[nz].double()).sum()), float((dv * sn[nz].double()).sum())))
        S["recs"][t]["pos"] = Ts_pos[-1]
    # the 2n single moves by angle bin: + moves of trit i point at al_i, - moves at al_i + pi
    al = torch.atan2(sn, cs) % (2 * math.pi)
    bp = (al * (NB / (2 * math.pi))).long().clamp_(0, NB - 1)
    bm = (bp + NB // 2) % NB
    del al, phi
    lists = []
    for b, d in ((bp, 1), (bm, -1)):
        order = torch.from_numpy(np.argsort(b.numpy().astype(np.uint16), kind="stable").astype(np.int32))
        cnt = torch.bincount(b, minlength=NB)
        offs = torch.cat([torch.zeros(1, dtype=torch.long), cnt.cumsum(0)])
        lists.append((order, offs, d))
    del bp, bm
    print(f"surface_map_big {TAG}: angle bins built, {time.time() - t0:.0f}s", flush=True)

    def sideways(v, psi, r):
        """the r legal single moves pointing closest to angle psi (bins outward from psi's bin); v CPU int8, edited"""
        b0 = int(psi * NB / (2 * math.pi)) % NB
        have, k = 0, 0
        while have < r and k <= NB // 2:
            w = max(1, k)                                # widen geometrically; gather the whole window fresh
            lo_b, hi_b = b0 - (k + w - 1), b0 + (k + w - 1)
            cand_i, cand_d, cand_a = [], [], []
            for order, offs, d in lists:
                segs = [(lo_b, hi_b)] if lo_b >= 0 and hi_b < NB else \
                       [(lo_b % NB, NB - 1), (0, hi_b % NB)] if lo_b < 0 else [(lo_b, NB - 1), (0, hi_b - NB)]
                for a_, z_ in segs:
                    ii = order[offs[a_]:offs[z_ + 1]].long()
                    vv = v[ii] + d
                    ok = (vv >= -1) & (vv <= 1)
                    ii = ii[ok]
                    ang = torch.atan2(sn[ii] * d, cs[ii] * d) % (2 * math.pi)
                    cand_i.append(ii); cand_d.append(torch.full((len(ii),), d, dtype=torch.int8)); cand_a.append(ang)
            ci = torch.cat(cand_i); cdd = torch.cat(cand_d); ca = torch.cat(cand_a)
            have = len(ci)
            k = k + w
        dd = (ca - psi + math.pi) % (2 * math.pi) - math.pi
        keep = dd.abs().argsort()[:r]
        ci, cdd = ci[keep], cdd[keep]
        v[ci] += cdd
        dx = float((cdd.double() * cs[ci].double()).sum()); dy = float((cdd.double() * sn[ci].double()).sum())
        return dx, dy, len(ci)

    P = torch.tensor(Ts_pos, dtype=torch.float64)
    lo, hi = P.min(0).values, P.max(0).values
    ctr, half = (lo + hi) / 2, float((hi - lo).max()) / 2 * (1 + 2 * MARGIN)
    xs = torch.linspace(float(ctr[0]) - half, float(ctr[0]) + half, N)
    ys = torch.linspace(float(ctr[1]) - half, float(ctr[1]) + half, N)
    Q = torch.stack(torch.meshgrid(xs, ys, indexing="xy"), -1).view(-1, 2).double()
    A, B = P[:-1], P[1:]
    AB = B - A
    f = (((Q[:, None] - A[None]) * AB[None]).sum(-1) / (AB * AB).sum(-1).clamp_min(1e-9)[None]).clamp(0, 1)
    dist = (Q[:, None] - (A[None] + f[..., None] * AB[None])).norm(dim=-1)
    ns = dist.shape[1]
    pad = torch.full((dist.shape[0], 1), float("inf"), dtype=dist.dtype)
    dl, dr = torch.cat([pad, dist[:, :-1]], 1), torch.cat([dist[:, 1:], pad], 1)
    locmin = (dist <= dl) & (dist <= dr) & (dist <= RMAX * half)
    locmin[torch.arange(len(Q)), dist.argmin(1)] = True
    cand = []
    for ci_ in range(len(Q)):
        ts = locmin[ci_].nonzero().flatten()
        kept = []
        for t in ts[dist[ci_, ts].argsort()].tolist():
            if all(abs(t - k) > 2 for k in kept):
                kept.append(t)
            if len(kept) == KMAX:
                break
        cand += [(ci_, t, float(f[ci_, t])) for t in kept]
    for t in range(ns):
        cand += [(-1, t, fr) for fr in FINE]
    nc = len(cand)
    c_cell = torch.tensor([c[0] for c in cand]); c_seg = torch.tensor([c[1] for c in cand])
    c_frac = torch.tensor([c[2] for c in cand]); c_loss = torch.full((nc,), float("nan"))
    c_x = torch.zeros(nc); c_y = torch.zeros(nc); c_mv = torch.zeros(nc)
    print(f"surface_map_big {TAG}: {nc} networks to build ({(c_cell >= 0).sum()} surface)", flush=True)
    for t in range(ns):
        idx = (c_seg == t).nonzero().flatten().tolist()
        if not idx:
            continue
        Ta = _trits(t, dev).cpu(); Tb = _trits(t + 1, dev).cpu()
        dif = Tb.to(torch.int16) - Ta.to(torch.int16)
        di = dif.nonzero().flatten(); dd_ = dif[di]
        g = torch.Generator().manual_seed(100 + t)
        u = torch.rand(2, len(di), generator=g)
        s1 = torch.where(dd_.abs() == 2, u.min(0).values, u[0]); s2 = u.max(0).values
        for j in idx:
            tm = [time.time()]
            fc = float(c_frac[j]); ci_ = int(c_cell[j])
            k = (fc >= s1).to(torch.int16) + ((dd_.abs() == 2) & (fc >= s2)).to(torch.int16)
            v = Ta.clone()
            step = (dd_.sign() * k)
            v[di] = (v[di].to(torch.int16) + step).to(torch.int8)
            # base position: evaluation t + the moves made
            st = step.double()
            px = Ts_pos[t][0] + float((st * cs[di].double()).sum()); py = Ts_pos[t][1] + float((st * sn[di].double()).sum())
            r = 0
            if ci_ >= 0:
                dx_, dy_ = float(Q[ci_, 0]) - px, float(Q[ci_, 1]) - py
                r = int(round(math.hypot(dx_, dy_)))
                if r > 0:
                    mx, my, r = sideways(v, math.atan2(dy_, dx_) % (2 * math.pi), r)
                    px += mx; py += my
            tm.append(time.time())
            model.load_state_dict(_floats(t, t + 1, fc), strict=False)
            tm.append(time.time())
            _set(L, v.to(dev))
            torch.cuda.synchronize(); tm.append(time.time())
            c_loss[j] = _orig(model, data, tc, device)
            tm.append(time.time())
            c_x[j], c_y[j], c_mv[j] = px, py, r
            if PROF:
                print(f"surface_map_big prof: build {tm[1]-tm[0]:.2f} floats {tm[2]-tm[1]:.2f} set {tm[3]-tm[2]:.2f} "
                      f"eval {tm[4]-tm[3]:.2f} (r {r})", flush=True)
        print(f"surface_map_big {TAG}: segment {t}: {len(idx)} networks, {time.time() - t0:.0f}s", flush=True)
    for r_ in S["recs"]:
        r_.pop("_", None)
    torch.save({"tag": TAG, "n": n, "N": N, "x": xs, "y": ys, "cell": c_cell, "seg": c_seg, "frac": c_frac,
                "loss": c_loss, "qx": c_x, "qy": c_y, "moves": c_mv, "recs": S["recs"], "version": 4}, OUT)
    print(f"surface_map_big {TAG}: wrote {OUT} ({nc} networks, {time.time() - t0:.0f}s)", flush=True)


@torch.no_grad()
def hook(model, data, tc, device):
    c = S["calls"]; S["calls"] += 1
    if S["L"] is None:
        names = {id(m): nm for nm, m in model.named_modules()}
        S["L"] = [m for m in model.modules() if isinstance(m, KernelTernaryLinear)]
        sd_keys = set(model.state_dict().keys())
        pk = []
        for m in S["L"]:
            nm = names[id(m)]
            pk.append(next(k for k in sd_keys if k == nm + ".wpacked"))
        S["pk"] = pk
        tr = {nm for nm, p in model.named_parameters() if p.requires_grad and p.is_floating_point()}
        S["fk"] = sorted(k for k in sd_keys if k in tr)
        print(f"surface_map_big {TAG}: {len(pk)} ternary layers, {len(S['fk'])} trainable float tensors "
              f"({sum(model.state_dict()[k].numel() for k in S['fk']) / 1e6:.1f}M)", flush=True)
    L = S["L"]
    it0 = tc.eval_iters; tc.eval_iters = ITERS
    sd = model.state_dict()
    if S["T0"] is None:
        S["T0"] = _get(L).cpu()
        if E("LS_T0"):
            try:
                st = torch.load(E("LS_T0"), map_location="cpu", weights_only=False)
                st = st.get("model", st)
                parts = []
                for k, m in zip(S["pk"], L):
                    kk = next(x for x in st if x.endswith(k.split("_orig_mod.")[-1]))
                    parts.append(unpack_rows(st[kk].to(device), m.K).flatten().cpu())
                S["T0"] = torch.cat(parts)
                print(f"surface_map_big {TAG}: origin from {E('LS_T0')}", flush=True)
            except Exception as e:
                print(f"surface_map_big {TAG}: LS_T0 not usable ({e!r}); origin = first evaluation", flush=True)
    loss0 = _orig(model, data, tc, device)
    snap = {k: sd[k].detach().cpu().clone() for k in S["pk"] + S["fk"]}
    S["snaps"].append(snap)
    T = _get(L).cpu()
    prevT = S.get("prevT")
    rec = {"call": c, "loss": loss0, "moves_from_start": float((T != S["T0"]).sum() if False else
                                                             (T.to(torch.int16) - S["T0"].to(torch.int16)).abs().sum()),
           "moves_since_prev": float((T.to(torch.int16) - prevT.to(torch.int16)).abs().sum()) if prevT is not None else 0.0}
    S["prevT"] = T
    S["recs"].append(rec)
    print(f"surface_map_big {TAG}: call {c} loss {loss0:.4f} moves {rec['moves_from_start']:.0f}", flush=True)
    if c == LAST - 1:
        cur = {k: v.clone() for k, v in snap.items()}
        grid(model, data, tc, device)
        model.load_state_dict({k: v.to(device) for k, v in cur.items()}, strict=False)
    tc.eval_iters = it0
    return loss0


TT.evaluate = hook
sys.argv = ["ttf.train"] + sys.argv[1:]
TT.main()
