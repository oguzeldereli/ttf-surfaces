"""basins_map.py (19 Nov 2026): several trained ternary networks of one shape on one plane, with the measured loss
surface across them and outward past them.
Plane: origin = the trit-wise median M of three networks A, B, C. Each trit gets one fixed unit vector:
  only X differs from M          -> along X's corner direction (90/210/330 deg, +-20 deg jitter), signed by X's move;
  X and Y differ (X=+1, Y=-1)     -> along dir_X - dir_Y (jitter), signed by X's move;
  all equal                       -> random angle.
So each network lands at its own corner, M at the centre. Networks on the plane are real ternary networks:
  inside the triangle: from M, take X's value on the trits where only X differs (or X/Y differ) with a fixed random
    per-trit draw r_i < w_X (Y: r_i > 1 - w_Y), the two weights solving w_X P_X + w_Y P_Y = target over the sector;
    floats mixed with c_X = w_X + (1 - w_X - w_Y)/3;
  outside it: the network on the triangle's edge nearest in weights, then the fewest legal single moves (+-1, staying
    in {-1,0,1}) whose vectors point most along the remaining offset (the sideways shift of surface_map.py).
Every point is measured at its actual position (the draw makes it land near, not on, the target). Also measured:
the straight in-between paths A-B, B-C, C-A and M-X.
Run as a ttf.train wrapper with the shape's arguments; all work happens at the first evaluation, then it exits.
BM_NETS="A=path,B=path,C=path" BM_OUT=out.pt [BM_N=41 BM_EXT=1.7 BM_ITERS=8] python basins_map.py <train args>"""
import math, os, sys, time
sys.path.insert(0, os.getcwd())
import torch
import ttf.train as TT
from ttf.kernel import pack_rows, unpack_rows
from ttf.layers import KernelTernaryLinear

E = os.environ.get
NETS = [kv.split("=", 1) for kv in E("BM_NETS").split(",")]
SAVE, SAVE_TPL = E("BM_SAVE"), E("BM_SAVE_TPL")
SAVEF = [float(v) for v in E("BM_SAVEF", "0.25,0.5,0.75").split(",")]
OUT, N, EXT, ITERS = E("BM_OUT"), int(E("BM_N", "41")), float(E("BM_EXT", "1.7")), int(E("BM_ITERS", "8"))
_orig = TT.evaluate


def strip(k):
    return k.split("_orig_mod.")[-1]


@torch.no_grad()
def run(model, data, tc, device):
    t0 = time.time()
    tc.eval_iters = ITERS
    L = [m for m in model.modules() if isinstance(m, KernelTernaryLinear)]
    msd = model.state_dict()
    name_of = {strip(k): k for k in msd}
    packed = {k for k in msd if k.endswith(".wpacked")}
    fkeys = [k for k in msd if k not in packed and msd[k].is_floating_point()]
    states, Ts = [], []
    for nm, p in NETS:
        st = torch.load(p, map_location="cpu", weights_only=False)["model"]
        st = {name_of[strip(k)]: v for k, v in st.items() if strip(k) in name_of}
        miss = set(msd) - set(st)
        assert not miss, f"{nm}: missing {sorted(miss)[:5]}"
        model.load_state_dict({k: v.to(device) for k, v in st.items()})
        Ts.append(torch.cat([unpack_rows(m.wpacked, m.K).flatten() for m in L]).to(torch.int8))
        states.append(st)
    keep = {k: v.clone() for k, v in states[0].items()}
    TT_ = torch.stack(Ts).to(torch.int16)                       # 3 x n
    n = TT_.shape[1]
    M = TT_.median(0).values
    dev = TT_ - M                                                # deviations from the median
    nz = dev != 0
    cnt = nz.sum(0)
    g = torch.Generator(device="cpu").manual_seed(11)
    rnd = lambda: torch.rand(n, generator=g, dtype=torch.float64).to(device)
    cor = [math.radians(a) for a in (90, 210, 330)]
    ang = rnd() * 2 * math.pi
    sgn = torch.ones(n, dtype=torch.float64, device=device)
    jit = (rnd() - 0.5) * math.radians(40)
    owner = torch.full((n,), -1, dtype=torch.long, device=device)   # single deviator
    pa = torch.full((n,), -1, dtype=torch.long, device=device)      # pair (X the +1 one, Y the -1 one)
    pb = torch.full((n,), -1, dtype=torch.long, device=device)
    for x in range(3):
        s = (cnt == 1) & nz[x]
        ang[s] = cor[x] + jit[s]; sgn[s] = dev[x][s].sign().double(); owner[s] = x
    for x in range(3):
        for y in range(3):
            if x == y:
                continue
            s = (cnt == 2) & (dev[x] > 0) & (dev[y] < 0)
            v = (math.cos(cor[x]) - math.cos(cor[y]), math.sin(cor[x]) - math.sin(cor[y]))
            ang[s] = math.atan2(v[1], v[0]) + jit[s]; pa[s] = x; pb[s] = y
    cs, sn = (sgn * ang.cos()).float(), (sgn * ang.sin()).float()
    r_i = rnd().float()

    def pos(T):
        d = (T.to(torch.int16) - M).float()
        return (float((d * cs).sum()), float((d * sn).sum()))

    P = [pos(TT_[x]) for x in range(3)]
    print(f"basins: n {n}, single {int((cnt == 1).sum())}, pairs {int((cnt == 2).sum())}, corners {P}", flush=True)

    def mixed2(w):                                               # pair trits: X (r<w_X) wins over Y (r>1-w_Y)
        T = M.clone()
        wt = torch.tensor(w, device=device, dtype=torch.float32)
        for x in range(3):
            s = (owner == x) & (r_i < w[x]); T[s] = TT_[x][s]
        hasp = pa >= 0
        wa, wb = wt[pa.clamp_min(0)], wt[pb.clamp_min(0)]
        sa = hasp & (r_i < wa); sb = hasp & ~sa & (r_i > 1 - wb)
        T[sa] = TT_[pa[sa], sa.nonzero().squeeze(1)]
        T[sb] = TT_[pb[sb], sb.nonzero().squeeze(1)]
        return T

    FS = float(E("BM_FSHARP", "1"))

    def floats(c):
        c = [max(v, 0.0) ** FS for v in c]; t = sum(c); c = [v / t for v in c]
        return {k: sum(c[x] * states[x][k].float() for x in range(3)).to(states[0][k].dtype) for k in fkeys}

    def evaluate(T, c):
        sd = dict(keep); sd.update(floats(c))
        model.load_state_dict({k: v.to(device) for k, v in sd.items()}, strict=False)
        o = 0
        for m in L:
            n_, K = m.wpacked.shape[0], m.K
            m.wpacked.copy_(pack_rows(T[o:o + n_ * K].view(n_, K).to(torch.int8))); o += n_ * K
        return _orig(model, data, tc, device)

    # legal single moves for the outward shift
    al = torch.atan2(sn, cs)

    def shift(T, r):
        rr = math.hypot(*r)
        if rr < 1:
            return T
        u = (r[0] / rr, r[1] / rr)
        proj = cs * u[0] + sn * u[1]                            # along +e_i: a +1 move; -e_i: a -1 move
        up = torch.where(T < 1, proj, torch.full_like(proj, -9))
        dn = torch.where(T > -1, -proj, torch.full_like(proj, -9))
        allp = torch.cat([up, dn])
        v, idx = allp.sort(descending=True)
        cum = v.double().cumsum(0)
        k = int(torch.searchsorted(cum, torch.tensor(rr, device=device, dtype=torch.float64))) + 1
        sel = idx[:k]
        T = T.clone()
        pl = sel[sel < n]; mi = sel[sel >= n] - n
        T[pl] += 1; T[mi] -= 1
        return T

    def wsolve(q):
        best = None
        for x in range(3):
            y = (x + 1) % 3
            A = torch.tensor([[P[x][0], P[y][0]], [P[x][1], P[y][1]]], dtype=torch.float64)
            w = torch.linalg.solve(A, torch.tensor(q, dtype=torch.float64))
            if w.min() >= -1e-9:
                best = (x, y, float(w[0]), float(w[1]))
        return best

    rec = {"name": [nm for nm, _ in NETS], "corner": P, "x": [], "y": [], "loss": [], "kind": [], "moves": [], "wt": []}

    def add(T, c, kind, w):
        l = evaluate(T, c)
        p = pos(T)
        rec["x"].append(p[0]); rec["y"].append(p[1]); rec["loss"].append(l); rec["kind"].append(kind)
        rec["moves"].append(int((T - M).abs().sum())); rec["wt"].append(list(w))
        return l

    if E("BM_LINES") is not None:
        x0 = int(E("BM_LINES"))
        rec.update({"line_from": x0, "frac": []})
        for y in range(3):
            if y == x0:
                continue
            for f in [i / 40 for i in range(0, 51)]:
                q = (P[x0][0] + f * (P[y][0] - P[x0][0]), P[x0][1] + f * (P[y][1] - P[x0][1]))
                x, yy, wx, wy = wsolve(q)
                w = [0.0] * 3; w[x] = wx; w[yy] = wy
                out = wx + wy > 1 + 1e-9
                if out:
                    s_ = wx + wy; w[x] /= s_; w[yy] /= s_
                T = mixed2(w)
                c = [ww + (1 - w[x] - w[yy]) / 3 for ww in w]
                if out:
                    b = pos(T); T = shift(T, (q[0] - b[0], q[1] - b[1]))
                add(T, c, f"line{y}", w); rec["frac"].append(f)
                if SAVE and any(abs(f - g_) < 1e-9 for g_ in SAVEF):    # the in-between network, as a resumable checkpoint
                    tpl = torch.load(SAVE_TPL, map_location="cpu", weights_only=False)
                    tpl["model"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; tpl["step"] = 0
                    os.makedirs(f"{SAVE}/l{x0}{y}_{f:.3f}", exist_ok=True)
                    torch.save(tpl, f"{SAVE}/l{x0}{y}_{f:.3f}/ckpt.pt")
            print(f"basins: line to {y} {time.time() - t0:.0f}s", flush=True)
        torch.save(rec, OUT); os._exit(0)
    if E("BM_RAY") is not None:
        x0 = int(E("BM_RAY"))
        NA, NR = int(E("BM_NA", "36")), int(E("BM_NR", "16"))
        dmax = 1.3 * max(math.hypot(P[y][0] - P[x0][0], P[y][1] - P[x0][1]) for y in range(3) if y != x0)
        radii = [2e4 * (dmax / 2e4) ** (i / (NR - 1)) for i in range(NR)]
        angs = [2 * math.pi * i / NA for i in range(NA)]
        angs += [math.atan2(P[y][1] - P[x0][1], P[y][0] - P[x0][0]) for y in range(3) if y != x0]
        angs = sorted(set(round(a_ % (2 * math.pi), 9) for a_ in angs))
        rec.update({"ray_from": x0, "angles": angs, "radii": radii, "ai": [], "ri": []})
        add(TT_[x0], [float(i == x0) for i in range(3)], f"net{x0}", [float(i == x0) for i in range(3)])
        rec["ai"].append(-1); rec["ri"].append(-1)
        for y in range(3):
            if y != x0:
                add(TT_[y], [float(i == y) for i in range(3)], f"net{y}", [float(i == y) for i in range(3)])
                rec["ai"].append(-1); rec["ri"].append(-1)
        for ai, a_ in enumerate(angs):
            for ri, r in enumerate(radii):
                q = (P[x0][0] + r * math.cos(a_), P[x0][1] + r * math.sin(a_))
                x, y, wx, wy = wsolve(q)
                w = [0.0] * 3; w[x] = wx; w[y] = wy
                out = wx + wy > 1
                if out:
                    s_ = wx + wy; w[x] /= s_; w[y] /= s_
                T = mixed2(w)
                c = [ww + (1 - w[x] - w[y]) / 3 for ww in w]
                if out:
                    b = pos(T); T = shift(T, (q[0] - b[0], q[1] - b[1]))
                add(T, c, "ray", w); rec["ai"].append(ai); rec["ri"].append(ri)
            print(f"basins: ray {ai + 1}/{len(angs)} {time.time() - t0:.0f}s", flush=True)
            torch.save(rec, OUT)
        rec["agree"] = [[float((TT_[a] == TT_[b]).float().mean()) for b in range(3)] for a in range(3)]
        rec["n"] = n
        torch.save(rec, OUT)
        print(f"basins: wrote {OUT} ({len(rec['loss'])} networks, {time.time() - t0:.0f}s)", flush=True)
        os._exit(0)
    # corners, centre, straight paths
    for x in range(3):
        add(TT_[x], [float(i == x) for i in range(3)], f"net{x}", [float(i == x) for i in range(3)])
    add(M, [1 / 3] * 3, "median", [0, 0, 0])
    F = [i / 20 for i in range(1, 20)]
    for x in range(3):
        y = (x + 1) % 3
        for f in F:
            w = [0.0] * 3; w[x] = 1 - f; w[y] = f
            add(mixed2(w), w, f"edge{x}{y}", w)
        for f in F:
            w = [0.0] * 3; w[x] = f
            c = [(1 - f) / 3] * 3; c[x] += f
            add(mixed2(w), c, f"spoke{x}", w)
    print(f"basins: paths done {time.time() - t0:.0f}s", flush=True)
    xs_ = [p[0] for p in P]; ys_ = [p[1] for p in P]
    half = max(max(xs_) - min(xs_), max(ys_) - min(ys_)) / 2 * EXT
    cx, cy = (max(xs_) + min(xs_)) / 2, (max(ys_) + min(ys_)) / 2
    for j in range(N):
        for i in range(N):
            q = (cx - half + 2 * half * i / (N - 1), cy - half + 2 * half * j / (N - 1))
            x, y, wx, wy = wsolve(q)
            w = [0.0] * 3; w[x] = wx; w[y] = wy
            out = wx + wy > 1
            if out:
                s = wx + wy; w[x] /= s; w[y] /= s
            T = mixed2(w)
            c = [ww + (1 - w[x] - w[y]) / 3 for ww in w]
            if out:
                b = pos(T); T = shift(T, (q[0] - b[0], q[1] - b[1]))
            add(T, c, "out" if out else "in", w)
        print(f"basins: row {j + 1}/{N} {time.time() - t0:.0f}s", flush=True)
    rec["agree"] = [[float((TT_[a] == TT_[b]).float().mean()) for b in range(3)] for a in range(3)]
    rec["n"] = n
    torch.save(rec, OUT)
    print(f"basins: wrote {OUT} ({len(rec['loss'])} networks, {time.time() - t0:.0f}s)", flush=True)
    os._exit(0)


def hook(model, data, tc, device):
    return run(model, data, tc, device)


TT.evaluate = hook
sys.argv = ["ttf.train"] + sys.argv[1:]
TT.main()
