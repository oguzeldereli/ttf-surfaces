"""agree.py: trit agreement between ternary checkpoints.
python agree.py REF1 REF2 ... [-- NET1 NET2 ...]: for each NET (default: the REFs) the share of trits equal to each
REF; with INIT=path also each NET's moved trits (vs INIT) and the overlap of moved sets between REFs."""
import os, sys, torch
from ttf.kernel import unpack_rows


def T(p):
    st = torch.load(p, map_location="cpu", weights_only=False)["model"]
    st = {k.split("_orig_mod.")[-1]: v for k, v in st.items()}
    out = []
    for k in sorted(st):
        if k.endswith(".wpacked"):
            pre, w = k[:-len(".wpacked")], st[k]
            K = st[pre + ".col_scale"].numel() if pre + ".col_scale" in st else w.shape[1] * 5
            out.append(unpack_rows(w, K).flatten().to(torch.int8))
    return torch.cat(out)


a = sys.argv[1:]
refs, nets = (a[:a.index("--")], a[a.index("--") + 1:]) if "--" in a else (a, a)
R = [T(p) for p in refs]
print("n trits", R[0].numel())
for p in nets:
    x = T(p)
    print("agree", os.path.relpath(p), " ".join(f"{(x == r).float().mean():.4f}" for r in R), flush=True)
if os.environ.get("INIT"):
    I = T(os.environ["INIT"])
    mv = [(r != I) for r in R]
    for i, p in enumerate(refs):
        print("moved", os.path.relpath(p), f"{mv[i].float().mean():.4f}")
    for i in range(len(R)):
        for j in range(i + 1, len(R)):
            both = (mv[i] & mv[j]); same = both & (R[i] == R[j])
            print(f"overlap {i}-{j}: moved in both {both.sum() / mv[i].sum():.4f} of {i}'s moves; same new value "
                  f"{same.sum() / both.sum().clamp_min(1):.4f}")
