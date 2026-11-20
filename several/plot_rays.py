"""plot_rays.py: basins_map.py BM_RAY outputs -> HTML. Rays out of one network in every direction of the plane, the
distance on a log scale, a clean angle x radius mesh; the exact rays toward the other networks drawn as lines and
plotted as loss against distance below.
python plot_rays.py OUT.html "Title=rays.pt" ["Title=rays.pt" ...]"""
import math, sys
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

out, maps = sys.argv[1], [a.split("=", 1) for a in sys.argv[2:]]
import os
fig = make_subplots(rows=2, cols=1, specs=[[{"type": "scene"}], [{"type": "xy"}]], row_heights=[0.7, 0.3], vertical_spacing=0.06,
                    subplot_titles=["", "loss along the straight line toward each other network, and past it"])
COL = ["#e8173b", "#1f77b4", "#2ca02c"]
groups, titles = [], []
for title, p in maps:
    d = torch.load(p, weights_only=False)
    x0 = d["ray_from"]; angs, radii = d["angles"], d["radii"]
    Z = np.array(d["loss"]); ai = np.array(d["ai"]); ri = np.array(d["ri"]); kind = np.array(d["kind"])
    NA, NR = len(angs), len(radii)
    G = np.full((NA, NR), np.nan)
    G[ai[ai >= 0], ri[ai >= 0]] = Z[ai >= 0]
    z0 = float(Z[kind == f"net{x0}"][0])
    R0 = radii[0] / 2
    rho = lambda r: math.log10(r / R0)
    xs, ys, zs = [0.0], [0.0], [z0]
    for a in range(NA):
        for r in range(NR):
            xs.append(rho(radii[r]) * math.cos(angs[a])); ys.append(rho(radii[r]) * math.sin(angs[a])); zs.append(G[a, r])
    idx = lambda a, r: 1 + a * NR + r
    I, J, K = [], [], []
    for a in range(NA):
        b = (a + 1) % NA
        I.append(0); J.append(idx(a, 0)); K.append(idx(b, 0))
        for r in range(NR - 1):
            I += [idx(a, r), idx(a, r + 1)]; J += [idx(b, r), idx(b, r)]; K += [idx(a, r + 1), idx(b, r + 1)]
    zs = np.array(zs)
    ht = ["start network"] + [f"{radii[r]:,.0f} moves out, angle {math.degrees(angs[a]):.0f}°<br>loss {G[a, r]:.3f}"
                              for a in range(NA) for r in range(NR)]
    g = []
    fig.add_trace(go.Mesh3d(x=xs, y=ys, z=zs, i=I, j=J, k=K, intensity=zs, colorscale="Viridis", opacity=0.95,
                            hovertext=ht, hoverinfo="text", showscale=False, flatshading=False, visible=False), row=1, col=1)
    g.append(len(fig.data) - 1)
    P = d["corner"]
    for y in range(3):
        if y == x0:
            continue
        ay = math.atan2(P[y][1] - P[x0][1], P[y][0] - P[x0][0]) % (2 * math.pi)
        a = int(np.argmin([abs((A - ay + math.pi) % (2 * math.pi) - math.pi) for A in angs]))
        dist = math.hypot(P[y][0] - P[x0][0], P[y][1] - P[x0][1])
        rr = [0.0] + [rho(r) for r in radii]
        fig.add_trace(go.Scatter3d(x=[v * math.cos(angs[a]) for v in rr], y=[v * math.sin(angs[a]) for v in rr],
                                   z=np.r_[z0, G[a]] + 0.05, mode="lines", line={"color": COL[y], "width": 7},
                                   name=f"ray toward {d['name'][y]}", visible=False), row=1, col=1)
        g.append(len(fig.data) - 1)
        zy = float(Z[kind == f"net{y}"][0])
        fig.add_trace(go.Scatter3d(x=[rho(dist) * math.cos(ay)], y=[rho(dist) * math.sin(ay)], z=[zy + 0.1], mode="markers+text",
                                   marker={"size": 7, "color": COL[y]}, text=[f"{d['name'][y]} ({zy:.2f}), {dist / 1e6:.1f}M moves"],
                                   showlegend=False, visible=False), row=1, col=1)
        g.append(len(fig.data) - 1)
        lp = p.replace("rays_", "lines_")
        if os.path.exists(lp):
            dl = torch.load(lp, weights_only=False)
            kk = np.array(dl["kind"]); fr = np.array(dl["frac"]); LZ = np.array(dl["loss"])
            sel = kk == f"line{y}"
            fig.add_trace(go.Scatter(x=fr[sel] * dist / 1e6, y=LZ[sel], mode="lines+markers", line={"color": COL[y]},
                                     name=f"straight toward {d['name'][y]}", visible=False), row=2, col=1)
        else:
            fig.add_trace(go.Scatter(x=[radii[0] / 2] + list(radii), y=np.r_[z0, G[a]], mode="lines+markers", line={"color": COL[y]},
                                     name=f"toward {d['name'][y]}", visible=False), row=2, col=1)
        g.append(len(fig.data) - 1)
        fig.add_trace(go.Scatter(x=[dist / 1e6, dist / 1e6], y=[3, 11], mode="lines", line={"color": COL[y], "dash": "dot"},
                                 showlegend=False, visible=False), row=2, col=1)
        g.append(len(fig.data) - 1)
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[z0 + 0.1], mode="markers+text", marker={"size": 7, "color": COL[x0]},
                               text=[f"{d['name'][x0]} ({z0:.2f})"], showlegend=False, visible=False), row=1, col=1)
    g.append(len(fig.data) - 1)
    groups.append(g)
    A = d["agree"]
    titles.append(f"{title}: rays out of {d['name'][x0]}; trits agree " +
                  ", ".join(f"{d['name'][a]}~{d['name'][b]} {A[a][b]:.0%}" for a, b in [(0, 1), (1, 2), (0, 2)]))
for t in groups[0]:
    fig.data[t].visible = True
btn = []
for gi, (title, _) in enumerate(maps):
    vis = [False] * len(fig.data)
    for t in groups[gi]:
        vis[t] = True
    btn.append({"label": title, "method": "update", "args": [{"visible": vis}, {"title": titles[gi]}]})
fig.update_layout(updatemenus=[{"buttons": btn, "x": 0.0, "y": 1.04, "xanchor": "left"}], title=titles[0], height=1250,
                  margin={"l": 0, "r": 0, "t": 70, "b": 0},
                  scene={"xaxis_title": "log10 distance (x)", "yaxis_title": "log10 distance (y)", "zaxis_title": "val loss",
                         "aspectmode": "manual", "aspectratio": {"x": 1.3, "y": 1.3, "z": 0.7}})
fig.update_xaxes(title_text="million moves from the start network (straight line, 40 steps per network distance)", row=2, col=1)
fig.update_yaxes(title_text="val loss", row=2, col=1)
GUIDE = """<div style="font-family:sans-serif;max-width:1100px;margin:12px 20px;line-height:1.45;font-size:14px">
<h2 style="margin:4px 0">Out of one basin into the next (15M)</h2>
<p>Three trained networks on one plane (same construction as the triangle map). From one of them, rays go out in 36
directions plus exactly toward each of the other two, 16 distances each from 20k moves to 1.3 times the farthest
network, every point a measured ternary network. Distance from the start network is drawn on a log scale (the basins
are narrow: about 0.5M moves wide, 3-5M moves apart), so the start basin is the bowl in the middle and the others show
as dips at the far rim along their rays. Floats follow the dominant network (mix weights to the 4th power).
Below: loss along the straight line toward each other network and 25% past it (linear distance); the dotted line is where that network sits.</p></div>"""
html = fig.to_html(include_plotlyjs=True, full_html=True)
open(out, "w").write(html.replace("<body>", "<body>" + GUIDE, 1))
print("wrote", out)
