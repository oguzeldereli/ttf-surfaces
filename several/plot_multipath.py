"""plot_multipath.py: multipath_map.py output -> interactive HTML: the measured surface under several paths on one
plane, each path on it, the fp's own loss above its rounded path, and loss against evaluation below.
python plot_multipath.py MAP.pt OUT.html "Heading" "guide paragraph" [ZMAX]"""
import sys
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

d = torch.load(sys.argv[1], weights_only=False)
out, head, guide = sys.argv[2], sys.argv[3], sys.argv[4]
zcap = float(sys.argv[5]) if len(sys.argv) > 5 else None
GAP = 4
c = {k: np.array(v) for k, v in d["cand"].items()}
N = d["N"]
COL = ["#e8173b", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf"]
cell, seg, L, X, Y = c["cell"], c["seg"], c["loss"], c["x"], c["y"]
nodeL = [nd["loss"] for P in d["paths"] for nd in P["nodes"]]
if zcap is None:
    zcap = max(nodeL) + 0.3
by = {}
for j in np.nonzero(cell >= 0)[0]:
    by.setdefault(int(cell[j]), []).append(int(j))
tris = set()
for a in range(N - 1):
    for b in range(N - 1):
        cs = [a * N + b, a * N + b + 1, (a + 1) * N + b, (a + 1) * N + b + 1]
        for ref in [j for q in cs for j in by.get(q, [])]:
            pick = [min(by[q], key=lambda j: abs(seg[j] - seg[ref])) if q in by else None for q in cs]
            for tri in ((0, 1, 2), (1, 3, 2)):
                v = [pick[i] for i in tri]
                if None not in v and seg[v].max() - seg[v].min() <= GAP:
                    tris.add(tuple(v))
T = np.array(sorted(tris))
sc = 1e3
Zs = np.minimum(L, zcap)
fig = make_subplots(rows=2, cols=1, specs=[[{"type": "scene"}], [{"type": "xy"}]], row_heights=[0.74, 0.26], vertical_spacing=0.05,
                    subplot_titles=["", "loss of each path's networks against evaluation"])
fig.add_trace(go.Mesh3d(x=X / sc, y=Y / sc, z=Zs, i=T[:, 0], j=T[:, 1], k=T[:, 2], intensity=Zs, colorscale="Viridis",
                        opacity=0.85, name="surface", showlegend=True, showscale=False,
                        hovertext=[f"loss {l:.3f}<br>{m:,} sideways moves<br>{d['paths'][p]['name']}, segment {t}"
                                   for l, m, p, t in zip(L, c["moves"], c["path"], c["t"])], hoverinfo="text"), row=1, col=1)
for pi, P in enumerate(d["paths"]):
    col = COL[pi % len(COL)]
    nd = P["nodes"]
    pts = [(t, nd[t]["pos"][0], nd[t]["pos"][1], nd[t]["loss"]) for t in range(len(nd))]
    s = (c["path"] == pi) & (cell < 0)
    pts += [(t + f, x, y, l) for t, f, x, y, l in zip(c["t"][s], c["frac"][s], X[s], Y[s], L[s])]
    pts = np.array(sorted(pts))
    fig.add_trace(go.Scatter3d(x=pts[:, 1] / sc, y=pts[:, 2] / sc, z=np.minimum(pts[:, 3], zcap) + 0.01, mode="lines",
                               line={"color": col, "width": 7}, name=P["name"], legendgroup=P["name"],
                               hovertext=[f"{P['name']}<br>loss {l:.3f}" for l in pts[:, 3]], hoverinfo="text"), row=1, col=1)
    fpL = [n_["rec"].get("loss") if "rounded_loss" in n_["rec"] else None for n_ in nd]
    if any(v is not None for v in fpL):
        k = [i for i, v in enumerate(fpL) if v is not None]
        fig.add_trace(go.Scatter3d(x=[nd[i]["pos"][0] / sc for i in k], y=[nd[i]["pos"][1] / sc for i in k],
                                   z=[min(fpL[i], zcap) for i in k], mode="lines", line={"color": col, "width": 4, "dash": "dash"},
                                   name=f"{P['name']}: the fp network's own loss", legendgroup=P["name"]), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=[nd[-1]["pos"][0] / sc], y=[nd[-1]["pos"][1] / sc], z=[min(nd[-1]["loss"], zcap) + 0.03],
                               mode="markers+text", marker={"size": 5, "color": col}, text=[f"{P['name']} {nd[-1]['loss']:.3f}"],
                               showlegend=False, legendgroup=P["name"]), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(range(len(nd))), y=[n_["loss"] for n_ in nd], mode="lines", line={"color": col},
                             name=P["name"], legendgroup=P["name"], showlegend=False), row=2, col=1)
    if any(v is not None for v in fpL):
        fig.add_trace(go.Scatter(x=list(range(len(nd))), y=fpL, mode="lines", line={"color": col, "dash": "dash"},
                                 legendgroup=P["name"], showlegend=False), row=2, col=1)
fig.layout.scene.update(xaxis_title="x (k moves)", yaxis_title="y (k moves)", zaxis_title="val loss",
                        zaxis={"range": [float(np.nanmin(L)) - 0.05, zcap + 0.05]},
                        aspectmode="manual", aspectratio={"x": 1.3, "y": 1.3, "z": 0.75})
fig.update_yaxes(title_text="val loss", row=2, col=1)
fig.update_xaxes(title_text="evaluation (each path from its own start)", row=2, col=1)
fig.update_layout(height=1350, margin={"l": 0, "r": 0, "t": 30, "b": 0}, legend={"x": 0.78, "y": 0.99})
info = "; ".join(f"{P['name']}: {P['own_movers']:,} trits moved only here, end {P['nodes'][-1]['loss']:.3f}" for P in d["paths"])
G = f"""<div style="font-family:sans-serif;max-width:1150px;margin:12px 20px;line-height:1.45;font-size:14px">
<h2 style="margin:4px 0">{head}</h2><p>{guide}</p><p style="color:#555">{info}. Surface: {len(L):,} measured networks
(heights above {zcap:.2f} drawn at {zcap:.2f}).</p></div>"""
open(out, "w").write(fig.to_html(include_plotlyjs=True, full_html=True).replace("<body>", "<body>" + G, 1))
print("wrote", out, len(T), "triangles")
