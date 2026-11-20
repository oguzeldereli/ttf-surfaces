"""multipath_map.py (19 Nov 2026): several training paths of one network shape on ONE plane, with the measured loss
surface under all of them (the v4 construction of surface_map.py, generalised to a tree of paths).

Paths: MP_SPEC json {"origin": ckpt or null, "paths": [{"name", "dir", "glob", "parent": null | "origin" | [p, i],
"stride", "sector": [deg_start, deg_width]}]}. Every node is a TTF checkpoint ({"model"[, "rec"]}); a path with a
parent starts at that node (it is prepended). fp networks enter as their converter rounding (conv_snaps.py), so
heights are always the measured loss of a ternary network; the fp's own loss rides along in "rec".
Projection: origin T0. Each trit gets one fixed unit vector: if it has a net move in exactly one path (end vs that
path's start), its angle lies in that path's sector, ordered by when it made its final move there, pointing along that
net move; trits with net moves in several paths or none get random angles. So every path draws out into its own sector.
Surface: a grid over all nodes; each grid point gets one network per nearby arm (local minimum of the distance over a
path's segments, within RMAX of the half-width, at most KMAX over all paths): the in-between network of that segment
(per-trit random move times, floats interpolated), then the fewest legal single moves sideways to the grid point.
Plus FINE in-between networks per segment (the drawn path). Every height is a measured loss on MP_ITERS fixed batches.
Run as a ttf.train wrapper with the ternary shape's arguments; everything happens at the first evaluation."""
import glob, json, math, os, sys, time
sys.path.insert(0, os.getcwd())
import torch
import ttf.train as TT
from ttf.kernel import pack_rows, unpack_rows
from ttf.layers import KernelTernaryLinear

E = os.environ.get
SPEC, OUT = json.load(open(E("MP_SPEC"))), E("MP_OUT")
N, ITERS = int(E("MP_N", "41")), int(E("MP_ITERS", "8"))
RMAX, KMAX, MARGIN = float(E("MP_RMAX", "0.5")), int(E("MP_KMAX", "3")), float(E("MP_MARGIN", "0.1"))
FINE = [0.25, 0.5, 0.75]
_orig = TT.evaluate
strip = lambda k: k.split("_orig_mod.")[-1]


def _window(C, c, r):
    tot = int(C[-1]); r = min(r, tot)
    base = int(C[c - 1]) if c > 0 else 0
    left = min(r // 2, base); right = min(r - left, tot - base); left = r - right
    lo = int(torch.searchsorted(C, torch.tensor(base - left + 1, device=C.device, dtype=C.dtype))) if left > 0 else c
    hi = int(torch.searchsorted(C, torch.tensor(base + right, device=C.device, dtype=C.dtype))) + 1 if right > 0 else c
    return min(lo, c), max(hi, c)


@torch.no_grad()
def run(model, data, tc, device):
    t0 = time.time()
    tc.eval_iters = ITERS
    L = [m for m in model.modules() if isinstance(m, KernelTernaryLinear)]
    msd = model.state_dict()
    own = {strip(k): k for k in msd}
    Ks = [m.K for m in L]
    lname = [n_ for n_, m in model.named_modules() if isinstance(m, KernelTernaryLinear)]
    pk = [own[strip(n_) + ".wpacked"] for n_ in lname]

    def load(f):
        d = torch.load(f, map_location="cpu", weights_only=False)
        st = {own[strip(k)]: v for k, v in d["model"].items() if strip(k) in own}
        T = torch.cat([unpack_rows(st[k], K).flatten().to(torch.int8) for k, K in zip(pk, Ks)])
        return {"st": st, "T": T, "rec": d.get("rec", {})}

    origin = load(SPEC["origin"]) if SPEC.get("origin") else None
    paths = []
    for p in SPEC["paths"]:
        fs = sorted(glob.glob(os.path.join(p["dir"], p.get("glob", "s*.pt"))))
        if p.get("drop_last"):
            fs = fs[:-p["drop_last"]]
        st_ = p.get("stride", 1)
        fs = fs[::st_] if st_ > 1 and len(fs) else fs
        nodes = [load(f) for f in fs]
        par = p.get("parent")
        if par == "origin":
            nodes = [origin] + nodes
        elif par is not None:
            nodes = [paths[par[0]]["nodes"][par[1]]] + nodes
        paths.append({"name": p["name"], "nodes": nodes, "sector": p["sector"]})
        print(f"multipath: path {p['name']}: {len(nodes)} nodes", flush=True)
    T0 = (origin or paths[0]["nodes"][0])["T"].to(device).to(torch.int16)
    n = T0.numel()

    # node losses (the same windows for everything)
    def put(st, T):
        model.load_state_dict({k: v.to(device) for k, v in st.items()}, strict=False)
        o = 0
        for m, K in zip(L, Ks):
            n_ = m.wpacked.shape[0]
            m.wpacked.copy_(pack_rows(T[o:o + n_ * K].view(n_, K).to(torch.int8))); o += n_ * K
    for P_ in paths:
        for nd in P_["nodes"]:
            if "loss" not in nd:
                put(nd["st"], nd["T"].to(device)); nd["loss"] = _orig(model, data, tc, device)
    print(f"multipath: node losses {time.time() - t0:.0f}s", flush=True)

    # projection
    g = torch.Generator(device="cpu").manual_seed(5)
    ang = (torch.rand(n, generator=g, dtype=torch.float64) * 2 * math.pi).to(device)
    sgn = torch.ones(n, dtype=torch.float64, device=device)
    owner = torch.full((n,), -1, dtype=torch.int32, device=device)
    cnt = torch.zeros(n, dtype=torch.int32, device=device)
    for pi, P_ in enumerate(paths):
        a = P_["nodes"][0]["T"].to(device).to(torch.int16); b = P_["nodes"][-1]["T"].to(device).to(torch.int16)
        mv = a != b
        cnt += mv.int(); owner = torch.where(mv, torch.full_like(owner, pi), owner)
    for pi, P_ in enumerate(paths):
        nodes = P_["nodes"]
        a = nodes[0]["T"].to(device).to(torch.int16); b = nodes[-1]["T"].to(device).to(torch.int16)
        tl = torch.full((n,), -1, dtype=torch.int32, device=device)
        prev = a
        for t, nd in enumerate(nodes[1:], 1):
            T = nd["T"].to(device).to(torch.int16)
            tl = torch.where(T != prev, torch.full_like(tl, t), tl); prev = T
        s = (cnt == 1) & (owner == pi)
        ns_ = int(s.sum())
        key = torch.where(s, tl.long() * n + torch.arange(n, device=device), torch.full((n,), 2 ** 62, device=device))
        rank = torch.empty(n, dtype=torch.float64, device=device)
        rank[key.argsort()] = torch.arange(n, dtype=torch.float64, device=device)
        s0, w0 = [math.radians(v) for v in P_["sector"]]
        ang = torch.where(s, s0 + w0 * rank / max(ns_, 1), ang)
        sgn = torch.where(s & (b < a), -1.0, sgn)
        P_["own_movers"] = ns_
        print(f"multipath: path {P_['name']}: {ns_} trits moved only here", flush=True)
    cs, sn = (sgn * ang.cos()).float(), (sgn * ang.sin()).float()
    del ang, sgn, owner, cnt

    def pos(T):
        dv = (T.to(device).to(torch.int16) - T0).float()
        return (float((dv * cs).sum()), float((dv * sn).sum()))
    for P_ in paths:
        for nd in P_["nodes"]:
            nd["pos"] = pos(nd["T"])
    # sideways move table
    al = torch.atan2(sn, cs) % (2 * math.pi)
    dirs = torch.cat([al, (al + math.pi) % (2 * math.pi)]); del al
    order = dirs.argsort(); dsorted = dirs[order]; del dirs
    pdel = torch.where(order < n, 1, -1).to(torch.int16)
    pidx = (order % n).int(); del order
    # grid and candidates
    allp = torch.tensor([nd["pos"] for P_ in paths for nd in P_["nodes"]], dtype=torch.float64)
    lo, hi = allp.min(0).values, allp.max(0).values
    ctr, half = (lo + hi) / 2, float((hi - lo).max()) / 2 * (1 + 2 * MARGIN)
    xs = torch.linspace(float(ctr[0]) - half, float(ctr[0]) + half, N)
    ys = torch.linspace(float(ctr[1]) - half, float(ctr[1]) + half, N)
    Q = torch.stack(torch.meshgrid(xs, ys, indexing="xy"), -1).view(-1, 2).double()
    segs, off = [], 0                      # (path, t, global id)
    for pi, P_ in enumerate(paths):
        for t in range(len(P_["nodes"]) - 1):
            segs.append((pi, t, off + t))
        P_["off"] = off; off += len(P_["nodes"]) + 10
    A_ = torch.tensor([paths[p]["nodes"][t]["pos"] for p, t, _ in segs], dtype=torch.float64)
    B_ = torch.tensor([paths[p]["nodes"][t + 1]["pos"] for p, t, _ in segs], dtype=torch.float64)
    AB = B_ - A_
    f = (((Q[:, None] - A_[None]) * AB[None]).sum(-1) / (AB * AB).sum(-1).clamp_min(1e-9)[None]).clamp(0, 1)
    dist = (Q[:, None] - (A_[None] + f[..., None] * AB[None])).norm(dim=-1)
    sp = torch.tensor([s[0] for s in segs])
    inf = torch.full((dist.shape[0], 1), float("inf"), dtype=dist.dtype)
    dl = torch.cat([inf, dist[:, :-1]], 1); dr = torch.cat([dist[:, 1:], inf], 1)
    same_l = torch.cat([torch.tensor([False]), sp[1:] == sp[:-1]]); same_r = torch.cat([sp[1:] == sp[:-1], torch.tensor([False])])
    dl[:, ~same_l] = float("inf"); dr[:, ~same_r] = float("inf")
    locmin = (dist <= dl) & (dist <= dr) & (dist <= RMAX * half)
    locmin[torch.arange(len(Q)), dist.argmin(1)] = True
    cand = []
    for ci in range(len(Q)):
        js = locmin[ci].nonzero().flatten()
        kept = []
        for j in js[dist[ci, js].argsort()].tolist():
            if all(segs[j][0] != segs[k][0] or abs(segs[j][1] - segs[k][1]) > 2 for k in kept):
                kept.append(j)
            if len(kept) == KMAX:
                break
        cand += [(ci, j, float(f[ci, j])) for j in kept]
    for j in range(len(segs)):
        cand += [(-1, j, fr) for fr in FINE]
    nc = len(cand)
    print(f"multipath: {nc} networks to build ({sum(c[0] >= 0 for c in cand)} surface), {time.time() - t0:.0f}s", flush=True)
    res = {k: [] for k in ("cell", "seg", "path", "t", "frac", "loss", "x", "y", "moves")}
    by = {}
    for c in cand:
        by.setdefault(c[1], []).append(c)
    for j, (pi, t, gid) in enumerate(segs):
        if j not in by:
            continue
        na, nb = paths[pi]["nodes"][t], paths[pi]["nodes"][t + 1]
        sa, sb = na["st"], nb["st"]
        Ta, Tb = na["T"].to(device).to(torch.int16), nb["T"].to(device).to(torch.int16)
        dif = Tb - Ta
        gg = torch.Generator(device="cpu").manual_seed(100 + gid)
        u = torch.rand(2, n, generator=gg).to(device)
        s1 = torch.where(dif.abs() == 2, u.min(0).values, u[0]); s2 = u.max(0).values
        for ci, _, fc in by[j]:
            k = (fc >= s1).to(Ta.dtype) + ((dif.abs() == 2) & (fc >= s2)).to(Ta.dtype)
            v = Ta + dif.sign() * k
            r = 0
            if ci >= 0:
                p0 = pos(v)
                d = (float(Q[ci][0]) - p0[0], float(Q[ci][1]) - p0[1])
                r = int(round(math.hypot(*d)))
                if r > 0:
                    psi = math.atan2(d[1], d[0]) % (2 * math.pi)
                    c = min(int(torch.searchsorted(dsorted, torch.tensor(psi, device=device, dtype=dsorted.dtype))), 2 * n - 1)
                    nv = v[pidx] + pdel
                    lg = (nv >= -1) & (nv <= 1); del nv
                    C = lg.int().cumsum(0, dtype=torch.int32)
                    lo_, hi_ = _window(C, c, r)
                    sel = torch.arange(lo_, hi_, device=device)[lg[lo_:hi_]]
                    v = v.clone(); v[pidx[sel].long()] += pdel[sel]
            st = {kk: (sa[kk].float() * (1 - fc) + sb[kk].float() * fc).to(sa[kk].dtype) if sa[kk].is_floating_point() else sa[kk]
                  for kk in sa}
            put(st, v)
            l = _orig(model, data, tc, device)
            px = pos(v)
            for kk, vv in zip(("cell", "seg", "path", "t", "frac", "loss", "x", "y", "moves"), (ci, gid, pi, t, fc, l, px[0], px[1], r)):
                res[kk].append(vv)
        print(f"multipath: {paths[pi]['name']} segment {t}: {len(by[j])} networks, {time.time() - t0:.0f}s", flush=True)
    out = {"N": N, "n": n, "xs": xs, "ys": ys, "cand": res, "version": "multipath1",
           "paths": [{"name": P_["name"], "off": P_["off"], "own_movers": P_["own_movers"], "sector": P_["sector"],
                      "nodes": [{"pos": nd["pos"], "loss": nd["loss"], "rec": nd["rec"]} for nd in P_["nodes"]]} for P_ in paths]}
    torch.save(out, OUT)
    print(f"multipath: wrote {OUT} ({nc} networks, {time.time() - t0:.0f}s)", flush=True)
    os._exit(0)


TT.evaluate = lambda model, data, tc, device: run(model, data, tc, device)
sys.argv = ["ttf.train"] + sys.argv[1:]
TT.main()
