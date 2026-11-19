# ttf-surfaces

Drawing the loss surface a ternary network trains down: the method and scripts of

> O. Guzeldereli. *Drawing the Loss Surface a Ternary Network Trains Down.* 2026.

A ternary network has an exact, discrete geometry: two networks are as far apart as the number of unit trit moves
($0 \leftrightarrow \pm 1$) between them. These scripts use it to draw the loss surface around a training path, with
**every point of the drawing a real network whose loss was measured**.

## The idea

**Projection.** Every trit $i$ gets one fixed unit vector $e_i$ in the plane. A $+1$ move is one step along $e_i$, a
$-1$ move one step the opposite way, and a network $T$ sits at $p(T) = \sum_i (T_i - T^0_i)\, e_i$. The angle of $e_i$
is set by *when* trit $i$ made its final move (1.5 turns over the run), pointing along its net move. So the moves of one
period of training add up instead of cancelling, and the path is drawn at about a fifth of its length in moves,
turning as training goes on. (Directions by storage order, or the path's main directions, lose almost all of it.)

**Surface.** For each grid point and each nearby arm of the path:

1. take the real network part of the way between two consecutive evaluations: each trit that changed between them makes
   its move at its own random time, floats are interpolated;
2. move it sideways to the grid point with the fewest moves possible: the legal single moves whose directions are
   closest to the direction still needed (each adds almost a full unit);
3. measure its loss on fixed validation batches. That is the height.

Going sideways means doing more, or undoing, the moves training made in one period, so the walls you see are the
measured cost of leaving the path. Where the path returns over ground it crossed earlier, each arm gets its own sheet,
drawn one above the other.

## Files

| file | what |
|---|---|
| `surface_map.py` | wraps the trainer; snapshots every evaluation; builds and measures the surface at the last one |
| `surface_map_big.py` | the same for models of ~1e9 trits (only changed tensors kept, positions updated move by move, angle bins) |
| `mid_split.py` | splits the in-between gain into trit and float parts |
| `plot_surface.py` | interactive HTML (plotly): layered surface, the path through measured in-between networks, flatness chart |
| `examples/` | example outputs |

## Use

The scripts wrap the TTF trainer ([ttf-ternary](https://github.com/oguzeldereli/ttf-ternary)); run them from its
directory with the trainer's own arguments.

```bash
# 15M conversion recovery at theta 16: 40 evaluations, the surface built at the 40th
LS_OUT=sf.pt LS_LAST=40 LS_T0=conv.pt python surface_map.py --recipe scratch-110m --preset s15 --resume \
    --steps 1000 --eval-interval 25 --ttf-theta 16 --ttf-theta-start 16 --ttf-theta-end 16 --out-dir run
python plot_surface.py surface.html "conversion=sf.pt"
```

Environment variables: `LS_OUT` (output), `LS_LAST` (evaluation at which the surface is built), `LS_N` (grid side,
41), `LS_ITERS` (validation batches per network, 4), `LS_T0` (checkpoint with the start trits), `LS_TURNS` (1.5),
`LS_KMAX` (sheets per grid point, 3), `LS_RMAX` (0.6), `LS_MARGIN` (0.12), `LS_ANGLE` (`time` or `index`).

Cost: a 41 x 41 grid with up to three sheets is about 3000 networks. That is about 15 minutes on a 16 GB GPU at 15M
parameters and about 45 minutes at 110M (31 x 31).

## License

PolyForm Noncommercial 1.0.0 (see `LICENSE`).
