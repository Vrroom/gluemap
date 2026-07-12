"""Test fold_graph on the LegoTractor valid_edges dump, with a before/after plot."""

import pickle
from pathlib import Path

import matplotlib
import networkx as nx
import numpy as np

from gluemap.utils.rigs import fold_graph

matplotlib.use("Agg")

REPO = Path(__file__).parent.parent
EDGES_PKL = Path(__file__).parent / "data" / "legotractor_valid_edges.pkl"
BEST_EDGES_PKL = Path(__file__).parent / "data" / "legotractor_best_edges.pkl"
PLOT_PATH = REPO / "Debug" / "fold_before_after.png"
BEST_PLOT_PATH = REPO / "Debug" / "best_fold_before_after.png"
N_FRAMES = 8
N_SENSORS = 3


def station_positions(n_frames: int, radius: float = 1.0) -> dict:
    pos = {}
    for n in range(n_frames):
        th = 2 * np.pi * n / n_frames
        pos[n] = np.array([radius * np.sin(th), radius * np.cos(th)])
    return pos


def plot_before_after(G: nx.Graph, F: nx.MultiGraph, path: Path) -> None:
    import matplotlib.pyplot as plt

    stations = station_positions(N_FRAMES)
    cmap = plt.get_cmap("tab10")
    pos = {
        8 * k + n: stations[n] + 0.18 * station_positions(N_SENSORS)[k]
        for k in range(N_SENSORS)
        for n in range(N_FRAMES)
    }
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 6.5))
    for i, j in G.edges():
        intra = i % N_FRAMES == j % N_FRAMES
        ax0.plot(*zip(pos[i], pos[j]),
                 color=cmap(i % N_FRAMES) if intra else "0.6",
                 lw=1.6 if intra else 0.7, alpha=0.9 if intra else 0.45, zorder=1)
    for idx in G.nodes():
        ax0.scatter(*pos[idx], s=90, color=cmap(idx % N_FRAMES), zorder=2)
        ax0.annotate(f"c{idx // N_FRAMES}f{idx % N_FRAMES}", pos[idx],
                     fontsize=6, ha="center", va="center", zorder=3)
    ax0.set_title(f"before: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    mult = {}
    for i, j in F.edges():
        mult[tuple(sorted((i, j)))] = mult.get(tuple(sorted((i, j))), 0) + 1
    for (i, j), m in mult.items():
        ax1.plot(*zip(stations[i], stations[j]), color="0.4", lw=0.8 + 0.55 * m, zorder=1)
        mid = (stations[i] + stations[j]) / 2
        ax1.annotate(str(m), mid, fontsize=10, ha="center", va="center",
                     bbox=dict(boxstyle="circle", fc="white", ec="0.4"), zorder=3)
    for n in F.nodes():
        ax1.scatter(*stations[n], s=650, color=cmap(n), zorder=2)
        ax1.annotate(f"f{n}", stations[n], fontsize=11, ha="center", va="center", zorder=3)
    ax1.set_title(f"after fold: {F.number_of_nodes()} nodes, "
                  f"{F.number_of_edges()} parallel edges, {len(mult)} pairs")
    for ax in (ax0, ax1):
        ax.set_aspect("equal")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def test_fold_valid_edges_structure_and_plot():
    assert EDGES_PKL.is_file(), f"(test_fold_graph): missing dump {EDGES_PKL}"
    G = nx.Graph()
    G.add_nodes_from(range(N_FRAMES * N_SENSORS))
    G.add_edges_from(pickle.load(open(EDGES_PKL, "rb")))
    BG = nx.Graph()
    BG.add_nodes_from(range(N_FRAMES * N_SENSORS))
    BG.add_edges_from(set(pickle.load(open(BEST_EDGES_PKL, "rb")).keys()))
    assert G.number_of_nodes() == 24, G.number_of_nodes()
    assert G.number_of_edges() == 65, G.number_of_edges()
    F = fold_graph(G, {idx: idx % N_FRAMES for idx in G.nodes})
    assert isinstance(F, nx.MultiGraph), type(F)
    assert F.number_of_nodes() == N_FRAMES, F.number_of_nodes()
    assert F.number_of_edges() == 41, F.number_of_edges()
    assert len({tuple(sorted(e)) for e in F.edges()}) == 7
    assert nx.is_tree(nx.Graph(F)), "folded graph is unexpectedly cyclic"
    intra = sum(1 for i, j in G.edges() if i % N_FRAMES == j % N_FRAMES)
    assert G.number_of_edges() - intra == F.number_of_edges(), (intra, F.number_of_edges())
    PLOT_PATH.parent.mkdir(exist_ok=True)
    plot_before_after(G, F, PLOT_PATH)
    plot_before_after(BG, F, BEST_PLOT_PATH)
    assert PLOT_PATH.is_file() and PLOT_PATH.stat().st_size > 10000

if __name__ == "__main__" : 
    test_fold_valid_edges_structure_and_plot()
