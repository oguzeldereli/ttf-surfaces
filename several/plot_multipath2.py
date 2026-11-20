"""plot_multipath2.py: multipath_map2.py output -> interactive HTML.
Every path has a complete sheet over the grid (its own networks, moved sideways to each grid point); the default view is
their lower envelope (at each grid point the lowest of the paths' networks), and buttons switch to one path's own sheet.
Paths are drawn on top from their measured in-between networks; below, loss against evaluation.
python plot_multipath2.py MAP.pt OUT.html "Heading" "guide paragraph" [ZMAX]"""
import sys
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

d = torch.load(sys.argv[1], weights_only=False)
out, head, guide = sys.argv[2], sys.argv[3], sys.argv[4]
zcap = float(sys.argv[5]) if len(sys.argv) > 5 else None
c = {k: np.array(v) for k, v in d["cand"].items()}
N, P = d["N"], d["paths"]
xs, ys = np.array(d["xs"]) / 1e3, np.array(d["ys"]) / 1e3
COL = ["#e8173b", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf"]
nodeL = [nd["loss"] for p in P for nd in p["nodes"]]
if zcap is None:
    zcap = max(nodeL) + 0.5
Z = np.full((len(P), N * N), np.nan)
for ci, pi, l in zip(c["cell"], c["path"], c["loss"]):
    if ci >= 0:
        Z[pi, ci] = l
Z = Z.reshape(len(P), N, N)                       # [path, y, x] (meshgrid "xy" order)
env = np.nanmin(Z, 0)
who = np.nanargmin(Z, 0)
fig = make_subplots(rows=2, cols=1, specs=[[{"type": "scene"}], [{"type": "xy"}]], row_heights=[0.74, 0.26],
                    vertical_spacing=0.05, subplot_titles=["", "loss of each path's networks against evaluation"])
cs = "Viridis"
names = [p["name"] for p in P]
fig.add_trace(go.Surface(x=xs, y=ys, z=np.minimum(env, zcap), surfacecolor=np.minimum(env, zcap), colorscale=cs,
                         cmin=float(np.nanmin(env)), cmax=zcap, showscale=False, opacity=0.9, name="lowest of all paths",
                         customdata=np.dstack([env, who]),
                         hovertemplate="loss %{customdata[0]:.3f}<extra>lowest here: path %{customdata[1]}</extra>"),
              row=1, col=1)
for pi in range(len(P)):
    fig.add_trace(go.Surface(x=xs, y=ys, z=np.minimum(Z[pi], zcap), surfacecolor=np.minimum(Z[pi], zcap), colorscale=cs,
                             cmin=float(np.nanmin(env)), cmax=zcap, showscale=False, opacity=0.9, visible=False,
                             name=f"sheet: {names[pi]}", hovertemplate=f"loss %{{z:.3f}}<extra>{names[pi]}</extra>"),
                  row=1, col=1)
nsurf = 1 + len(P)
for pi, p in enumerate(P):
    col = COL[pi % len(COL)]
    nd = p["nodes"]
    pts = [(t, nd[t]["pos"][0], nd[t]["pos"][1], nd[t]["loss"]) for t in range(len(nd))]
    s = (c["path"] == pi) & (c["cell"] < 0)
    pts += [(t + f, x, y, l) for t, f, x, y, l in zip(c["t"][s], c["frac"][s], c["x"][s], c["y"][s], c["loss"][s])]
    pts = np.array(sorted(pts))
    fig.add_trace(go.Scatter3d(x=pts[:, 1] / 1e3, y=pts[:, 2] / 1e3, z=np.minimum(pts[:, 3], zcap) + 0.01, mode="lines",
                               line={"color": col, "width": 7}, name=p["name"], legendgroup=p["name"],
                               hovertext=[f"{p['name']}<br>loss {l:.3f}" for l in pts[:, 3]], hoverinfo="text"), row=1, col=1)
    fpL = [n_["rec"].get("loss") if "rounded_loss" in n_["rec"] else None for n_ in nd]
    if any(v is not None for v in fpL):
        k = [i for i, v in enumerate(fpL) if v is not None]
        fig.add_trace(go.Scatter3d(x=[nd[i]["pos"][0] / 1e3 for i in k], y=[nd[i]["pos"][1] / 1e3 for i in k],
                                   z=[min(fpL[i], zcap) for i in k], mode="lines", line={"color": col, "width": 4, "dash": "dash"},
                                   name=f"{p['name']}: the fp network's own loss", legendgroup=p["name"]), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=[nd[-1]["pos"][0] / 1e3], y=[nd[-1]["pos"][1] / 1e3], z=[min(nd[-1]["loss"], zcap) + 0.03],
                               mode="markers+text", marker={"size": 5, "color": col}, text=[f"{nd[-1]['loss']:.3f}"],
                               showlegend=False, legendgroup=p["name"]), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(range(len(nd))), y=[n_["loss"] for n_ in nd], mode="lines", line={"color": col},
                             name=p["name"], legendgroup=p["name"], showlegend=False), row=2, col=1)
    if any(v is not None for v in fpL):
        fig.add_trace(go.Scatter(x=list(range(len(nd))), y=fpL, mode="lines", line={"color": col, "dash": "dash"},
                                 legendgroup=p["name"], showlegend=False), row=2, col=1)
ntr = len(fig.data)
btn = []
for b in range(nsurf):
    vis = [i == b for i in range(nsurf)] + [True] * (ntr - nsurf)
    btn.append({"label": "lowest of all paths" if b == 0 else f"sheet: {names[b - 1]}", "method": "restyle",
                "args": [{"visible": vis}]})
fig.update_layout(updatemenus=[{"buttons": btn, "direction": "down", "x": 0.0, "y": 1.0, "xanchor": "left",
                                "yanchor": "top", "showactive": True}])
fig.layout.scene.update(xaxis_title="x (k moves)", yaxis_title="y (k moves)", zaxis_title="val loss",
                        zaxis={"range": [float(np.nanmin(env)) - 0.05, zcap + 0.05]},
                        aspectmode="manual", aspectratio={"x": 1.3, "y": 1.3, "z": 0.75})
fig.update_yaxes(title_text="val loss", row=2, col=1)
fig.update_xaxes(title_text="evaluation (each path from its own start)", row=2, col=1)
fig.update_layout(height=1350, margin={"l": 0, "r": 0, "t": 30, "b": 0}, legend={"x": 0.78, "y": 0.99})
share = ", ".join(f"{names[pi]} {100 * np.mean(who == pi):.0f}%" for pi in range(len(P)))
info = "; ".join(f"{p['name']}: {p['own_movers']:,} trits moved only here, end {p['nodes'][-1]['loss']:.3f}" for p in P)
G = f"""<div style="font-family:sans-serif;max-width:1150px;margin:12px 20px;line-height:1.45;font-size:14px">
<h2 style="margin:4px 0">{head}</h2><p>{guide}</p>
<p><b>How the surface is made.</b> Every height is the measured loss of a real ternary network. For each grid point and
each path: take the network on that path nearest to the point (between two snapshots: each trit's move gets a random
time, floats interpolated), then make the fewest legal single-trit moves sideways until the network projects onto the
grid point. That gives every path its own complete sheet ({N}x{N} networks each). The default view is the lower
envelope: at each point the lowest of the paths' sheets. The menu at top left shows one path's own sheet; on its own
sheet a path lies exactly on the surface. Share of the envelope from each path: {share}.</p>
<p style="color:#555">{info}. Heights above {zcap:.2f} drawn at {zcap:.2f}.</p></div>"""
open(out, "w").write(fig.to_html(include_plotlyjs=True, full_html=True).replace("<body>", "<body>" + G, 1))
print("wrote", out)
