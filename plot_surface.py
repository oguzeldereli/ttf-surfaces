"""plot_surface.py: surface_map.py outputs (several networks per grid point, one per nearby arm of the path)
-> one interactive HTML: layered measured surface + the training path going down it. Several runs on one panel switch
with a menu (each run has its own coordinates: the vectors are set by that run's own move timing).
python plot_surface.py OUT.html "TITLE=a.pt[,b.pt,...]" ["TITLE2=c.pt" ...]"""
import sys
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

GAP = 4            # neighbouring networks join into one sheet when built from path segments at most GAP apart
items = [a.split("=", 1) for a in sys.argv[2:]]
specs, titles, heights = [], [], []
for t, p in items:
    specs.append([{"type": "scene", "colspan": 2}, None]); titles.append(t); heights.append(0.8)
    multi = "," in p and "theta" in p
    specs.append([{"type": "xy"}, {"type": "xy"} if multi else None])
    titles += ["is it flat? rise of each surface network above the path, against its sideways moves"]
    if multi:
        titles.append("path length and final loss against θ (same start, 1000 steps)")
    heights.append(0.3)
fig = make_subplots(rows=len(specs), cols=2, specs=specs, subplot_titles=titles, row_heights=heights,
                    vertical_spacing=0.035, horizontal_spacing=0.08)
PAL = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf"]


def name(tag):
    return tag.replace("conv_theta", "θ").replace("ternary_", "")


def build(d):
    """mesh triangles over the grid: at each grid square, for every network at a corner as reference, take at each
    other corner the network built from the closest path segment; a triangle where all are within GAP segments"""
    N = d["N"]
    cell, seg = d["cell"].numpy(), d["seg"].numpy()
    surf = np.nonzero(cell >= 0)[0]
    by = {}
    for j in surf:
        by.setdefault(int(cell[j]), []).append(int(j))
    tris = set()
    for a in range(N - 1):
        for b in range(N - 1):
            corners = [a * N + b, a * N + b + 1, (a + 1) * N + b, (a + 1) * N + b + 1]
            if not all(c in by for c in corners):
                pass
            refs = [j for c in corners for j in by.get(c, [])]
            for ref in refs:
                pick = []
                for c in corners:
                    js = by.get(c, [])
                    pick.append(min(js, key=lambda j: abs(seg[j] - seg[ref])) if js else None)
                for tri in ((0, 1, 2), (1, 3, 2)):
                    v = [pick[i] for i in tri]
                    if None in v:
                        continue
                    ss = seg[v]
                    if ss.max() - ss.min() <= GAP:
                        tris.add(tuple(v))
    return np.array(sorted(tris)) if tris else np.zeros((0, 3), int)


menus, row, sc_i = [], 1, 0
for title, paths in items:
    runs = [torch.load(p, map_location="cpu", weights_only=False) for p in paths.split(",")]
    runs.sort(key=lambda d: (float(d["tag"].split("theta")[-1].split("_s")[0]) if "theta" in d["tag"] else 0, d["tag"]))
    zmin = min(float(np.nanmin(d["loss"].numpy())) for d in runs)
    zmax = max(max(r["loss"] for r in d["recs"]) for d in runs) + 0.05
    groups = []
    for i, d in enumerate(runs):
        g0 = len(fig.data)
        L, X, Y = d["loss"].numpy(), d["qx"].numpy(), d["qy"].numpy()
        cell, seg, fr, mv = d["cell"].numpy(), d["seg"].numpy(), d["frac"].numpy(), d["moves"].numpy()
        # path: evaluation networks + the in-between networks (no sideways moves), in order
        recs = d["recs"]
        pts = [(t, r["pos"][0], r["pos"][1], r["loss"]) for t, r in enumerate(recs)]
        pts += [(seg[j] + fr[j], X[j], Y[j], L[j]) for j in np.nonzero(cell < 0)[0]]
        pts.sort()
        pt = np.array(pts)
        rise = L - np.interp(seg + fr, pt[:, 0], pt[:, 3])
        T = build(d)
        txt = [f"{name(d['tag'])}<br>loss {L[j]:.4f}<br>from the path between evals {seg[j]} and {seg[j] + 1} "
               f"({fr[j]:.2f} of the way), then {int(mv[j])} moves sideways<br>rise above the path {rise[j]:+.4f}"
               for j in range(len(L))]
        vis = i == 0
        fig.add_trace(go.Mesh3d(x=X, y=Y, z=np.minimum(L, zmax), i=T[:, 0], j=T[:, 1], k=T[:, 2],
                                intensity=np.minimum(L, zmax), colorscale="Viridis", cmin=zmin, cmax=zmax,
                                opacity=0.85, showscale=False, hovertext=txt, hoverinfo="text", visible=vis,
                                name=f"{name(d['tag'])} surface"), row=row, col=1)
        fig.add_trace(go.Scatter3d(x=pt[:, 1], y=pt[:, 2], z=pt[:, 3], mode="lines", visible=vis,
                                   name=f"{name(d['tag'])}: training path", line={"color": "#e8173b", "width": 7},
                                   hoverinfo="skip"), row=row, col=1)
        P = np.array([r["pos"] for r in recs]); Lz = np.array([r["loss"] for r in recs])
        fig.add_trace(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=Lz, mode="markers", visible=vis, showlegend=False,
                                   marker={"size": 3, "color": "#e8173b"},
                                   text=[f"{name(d['tag'])} eval {r['call']}<br>loss {r['loss']:.4f}<br>"
                                         f"{r['moves_from_start']:.0f} moves from the start" for r in recs],
                                   hoverinfo="text"), row=row, col=1)
        fig.add_trace(go.Scatter3d(x=[P[0, 0], P[-1, 0]], y=[P[0, 1], P[-1, 1]], z=[Lz[0] + 0.02, Lz[-1] + 0.02],
                                   mode="markers+text", text=["first eval", f"{name(d['tag'])} end"], visible=vis,
                                   showlegend=False, marker={"size": 6, "color": "black", "symbol": "diamond"}),
                      row=row, col=1)
        groups.append(list(range(g0, len(fig.data))))
        surfmask = cell >= 0
        fig.add_trace(go.Scatter(x=mv[surfmask], y=rise[surfmask], mode="markers", showlegend=False,
                                 marker={"size": 3, "color": PAL[i % len(PAL)], "opacity": 0.4}, name=name(d["tag"]),
                                 hovertemplate=name(d["tag"]) + "<br>%{x:.0f} moves sideways<br>rise %{y:+.4f}<extra></extra>"),
                      row=row + 1, col=1)
        print(f"{d['tag']}: {surfmask.sum()} surface networks, {len(T)} triangles, path {len(pt)} points")
    fig.update_xaxes(title_text="sideways moves from the path", row=row + 1, col=1)
    fig.update_yaxes(title_text="loss rise above the path", row=row + 1, col=1)
    sc = "scene" if sc_i == 0 else f"scene{sc_i + 1}"; sc_i += 1
    fig.layout[sc].update(xaxis_title="x (moves)", yaxis_title="y (moves)", zaxis_title="val loss",
                          zaxis={"range": [zmin - 0.02, zmax]}, aspectmode="manual",
                          aspectratio={"x": 1.3, "y": 1.3, "z": 0.8})
    if len(runs) > 1:
        allidx = [j for g in groups for j in g]
        btn = [{"label": f"run: {name(d['tag'])}", "method": "restyle",
                "args": [{"visible": [j in groups[i] for j in allidx]}, allidx]} for i, d in enumerate(runs)]
        dom = fig.layout[sc].domain.y
        menus.append({"buttons": btn, "direction": "down", "x": 0.0, "xanchor": "left", "y": dom[1], "yanchor": "top",
                      "active": 0, "showactive": True})
    if len(runs) > 1 and all("theta" in d["tag"] for d in runs):
        th = [float(d["tag"].split("theta")[-1].split("_s")[0]) for d in runs]
        plen = [sum(r["moves_since_prev"] for r in d["recs"]) + d["recs"][0]["moves_from_start"] for d in runs]
        straight = [d["recs"][-1]["moves_from_start"] for d in runs]
        fin = [d["recs"][-1]["loss"] for d in runs]
        fig.add_trace(go.Scatter(x=th, y=[p / s for p, s in zip(plen, straight)], mode="lines+markers", showlegend=False,
                                 line={"color": "#444"}, name="path / straight",
                                 text=[f"θ{t:g}: path {p:.3g} moves, straight {s:.3g}, final loss {f:.4f}"
                                       for t, p, s, f in zip(th, plen, straight, fin)],
                                 hovertemplate="%{text}<extra></extra>"), row=row + 1, col=2)
        fig.add_trace(go.Scatter(x=th, y=[(f - min(fin)) * 100 + 1 for f in fin], mode="lines+markers",
                                 showlegend=False, line={"color": "#d62728", "dash": "dash"}, name="final loss",
                                 text=[f"θ{t:g}: final loss {f:.4f}" for t, f in zip(th, fin)],
                                 hovertemplate="%{text}<extra></extra>"), row=row + 1, col=2)
        fig.update_xaxes(type="log", title_text="θ", row=row + 1, col=2)
        fig.update_yaxes(title_text="path / straight (solid); (final − best) x100 + 1 (dashed)", row=row + 1, col=2)
    row += 2
fig.update_layout(height=sum(1100 * h for h in heights), margin={"l": 0, "r": 0, "t": 40, "b": 0},
                  legend={"x": 0.8, "y": 1.0}, updatemenus=menus)
GUIDE = """<div style="font-family:sans-serif;max-width:1100px;margin:12px 20px;line-height:1.45;font-size:14px">
<h2 style="margin:4px 0">The loss surface the training path goes down (15M ternary, one vector per trit)</h2>
<p><b>Coordinates</b>: every trit has its own fixed unit vector, a +1 move one step along it, a -1 move one step the
opposite way; a network's position is the sum of its steps from the start network. The angle of a trit's vector is set
by <i>when</i> it made its final move in the run (1.5 turns from the start of training to the end), pointing along its
net move, so the moves of each period add up: the path is drawn at its real length in moves and turns as training goes
on. Trits that end where they started get random angles. Each run has its own coordinates (set by its own move timing),
so runs are shown one at a time (menu). Height: measured validation loss.</p>
<p><b>The surface</b>: every point is a real network with its measured loss. For each grid point and each nearby arm of
the path, the network starts from the nearest point of that arm (the real network that far between two evaluations:
that share of the moves actually made between them, float parts blended the same way) and moves sideways to the grid
point with the fewest moves possible (the legal single moves pointing closest to that direction). Where two arms lie
side by side, the same spot is built from both: each sheet continues past the other arm and rises as it leaves its own
path, one sheet above the other (the layers). <b>Red line</b>: the training path through the evaluation networks
(dots) and four in-between networks per interval, so it runs on the surface.</p>
<p><b>Lower left</b>: rise of every surface network above the path point it started from, against its sideways moves.
<b>Lower right</b>: how much longer each conversion path is than the straight distance from start to end (moves
between evaluations every 25 steps; reversals within 25 steps are not counted), and the final loss.</p>
</div>"""
html = fig.to_html(include_plotlyjs=True, full_html=True)
open(sys.argv[1], "w").write(html.replace("<body>", "<body>" + GUIDE, 1))
print("wrote", sys.argv[1])
