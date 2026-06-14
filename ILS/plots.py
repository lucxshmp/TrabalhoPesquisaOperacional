"""
Gráficos do ILS-RVND (T5.2–T5.4), gerados a partir dos CSVs do experiments.py.

  - T5.2 convergência: custo da melhor × tempo decorrido, 1 linha por execução,
    com a mediana (interpolada em grade comum) destacada — 1 PNG por instância.
  - T5.3 boxplot: distribuição do custo final das N execuções por instância.
  - T5.4 mapa de rotas: scatter lat/lon dos clientes + depósito, com as rotas
    coloridas — "antes" (ida-e-volta) vs. "depois" (solução ILS), estilo Fig. 4.

Lê: convergencia.csv, experimentos_execucoes.csv, melhor_rotas.csv
(produzidos por experiments.run_and_save) e a própria Instance (coordenadas).

Uso:
    python plots.py                 # usa resultados/ e datasets padrão
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")          # backend sem display (WSL/headless)
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Leitura dos CSVs
# ---------------------------------------------------------------------------

def read_convergence(path):
    """convergencia.csv → {instancia: [run0, run1, ...]} com run = [(t, custo)]."""
    por_inst = defaultdict(lambda: defaultdict(list))
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            por_inst[row["instancia"]][int(row["run"])].append(
                (float(row["tempo_s"]), float(row["custo"]))
            )
    return {inst: [runs[k] for k in sorted(runs)] for inst, runs in por_inst.items()}


def read_runs(path):
    """experimentos_execucoes.csv → {instancia: [custo_final, ...]}."""
    por_inst = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            por_inst[row["instancia"]].append(float(row["custo"]))
    return dict(por_inst)


def read_routes(path):
    """melhor_rotas.csv → {instancia: [[no, ...], ...]} (rotas ordenadas)."""
    por_inst = defaultdict(lambda: defaultdict(list))
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            por_inst[row["instancia"]][int(row["rota"])].append(
                (int(row["posicao"]), int(row["no"]))
            )
    saida = {}
    for inst, rotas in por_inst.items():
        saida[inst] = [[no for _, no in sorted(rotas[k])] for k in sorted(rotas)]
    return saida


# ---------------------------------------------------------------------------
# T5.2 — convergência
# ---------------------------------------------------------------------------

def _median_curve(runs, n_grid=200):
    """Mediana do custo (best-so-far, função-escada) sobre uma grade de tempo comum."""
    t_end = max(r[-1][0] for r in runs)
    grid = np.linspace(0.0, t_end, n_grid)
    mat = []
    for r in runs:
        ts = np.array([p[0] for p in r])
        cs = np.array([p[1] for p in r])
        idx = np.clip(np.searchsorted(ts, grid, side="right") - 1, 0, len(cs) - 1)
        vals = cs[idx].astype(float)
        vals[grid < ts[0]] = cs[0]      # antes do 1º registro: custo inicial
        mat.append(vals)
    return grid, np.median(np.array(mat), axis=0)


def plot_convergence(conv_by_inst, output_dir):
    """Uma figura de convergência por instância. Retorna os caminhos dos PNGs."""
    paths = []
    for inst, runs in conv_by_inst.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, r in enumerate(runs):
            ts = [p[0] for p in r]
            cs = [p[1] for p in r]
            ax.step(ts, cs, where="post", color="0.7", linewidth=1,
                    label="execuções" if i == 0 else None)
        grid, med = _median_curve(runs)
        ax.plot(grid, med, color="C3", linewidth=2.5, label="mediana")
        ax.set_xlabel("tempo decorrido (s)")
        ax.set_ylabel("custo da melhor solução (km)")
        ax.set_title(f"Convergência do ILS-RVND — {inst} ({len(runs)} execuções)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        path = os.path.join(output_dir, f"convergencia_{inst}.png")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# T5.3 — boxplot
# ---------------------------------------------------------------------------

def plot_boxplot(costs_by_inst, output_dir, fname="boxplot_custo.png"):
    """Boxplot do custo final das N execuções, uma caixa por instância."""
    labels = list(costs_by_inst.keys())
    dados = [costs_by_inst[k] for k in labels]
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(labels)), 5))
    ax.boxplot(dados, tick_labels=labels, showmeans=True)
    ax.set_ylabel("custo final (km)")
    ax.set_title("Distribuição do custo final por instância (N execuções)")
    ax.grid(True, axis="y", alpha=0.3)
    path = os.path.join(output_dir, fname)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# T5.4 — mapa de rotas
# ---------------------------------------------------------------------------

def _draw_routes(ax, instance, routes, titulo):
    lats = [nd.latitude for nd in instance.nodes]
    lons = [nd.longitude for nd in instance.nodes]
    # clientes
    ax.scatter(lons[1:], lats[1:], s=10, color="0.4", zorder=3)
    # depósito
    ax.scatter([lons[0]], [lats[0]], s=180, marker="*", color="black",
               zorder=5, label="depósito")
    cmap = plt.get_cmap("hsv", max(len(routes), 1) + 1)
    for ri, route in enumerate(routes):
        if not route:
            continue
        xs = [lons[0]] + [lons[n] for n in route] + [lons[0]]
        ys = [lats[0]] + [lats[n] for n in route] + [lats[0]]
        ax.plot(xs, ys, color=cmap(ri), linewidth=1, alpha=0.8, zorder=2)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(titulo)
    ax.grid(True, alpha=0.3)


def plot_routes(instance, routes_depois, output_dir, instancia,
                routes_antes=None, fname=None):
    """Mapa 'antes' (ida-e-volta) vs. 'depois' (ILS). Retorna o caminho do PNG."""
    if routes_antes is None:
        routes_antes = [[i] for i in range(1, len(instance.nodes))]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    _draw_routes(ax1, instance, routes_antes,
                 f"Antes: ida-e-volta ({len(routes_antes)} rotas)")
    n_dep = sum(1 for r in routes_depois if r)
    _draw_routes(ax2, instance, routes_depois,
                 f"Depois: ILS-RVND ({n_dep} rotas)")
    ax1.legend(loc="best")
    fig.suptitle(f"Rotas — {instancia}")
    path = os.path.join(output_dir, fname or f"mapa_rotas_{instancia}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    import argparse
    ils_dir = os.path.dirname(__file__)
    p = argparse.ArgumentParser(description="Gráficos do ILS-RVND (T5.2–T5.4)")
    p.add_argument("--output-dir", default=os.path.join(ils_dir, "resultados"))
    p.add_argument("--dados", default=os.path.join(ils_dir, "datasets", "dadosPO.xlsx"))
    p.add_argument("--depot-lat", type=float, default=-19.9)
    p.add_argument("--depot-lon", type=float, default=-44.1)
    p.add_argument("--Q", type=int, default=100)
    p.add_argument("--Y", type=float, default=8.0)
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--max-clientes", type=int, default=300)
    return p.parse_args()


def main():
    from instance_reader import load_instance
    args = _parse_args()
    out = args.output_dir

    conv = read_convergence(os.path.join(out, "convergencia.csv"))
    runs = read_runs(os.path.join(out, "experimentos_execucoes.csv"))
    rotas = read_routes(os.path.join(out, "melhor_rotas.csv"))

    print("Gerando convergência...")
    for p in plot_convergence(conv, out):
        print(f"  {p}")
    print("Gerando boxplot...")
    print(f"  {plot_boxplot(runs, out)}")

    print("Gerando mapa de rotas...")
    instance = load_instance(
        path=args.dados, depot_lat=args.depot_lat, depot_lon=args.depot_lon,
        Q=args.Q, Y=args.Y, K=args.K, max_clientes=(args.max_clientes or None),
    )
    for inst, routes in rotas.items():
        print(f"  {plot_routes(instance, routes, out, inst)}")


if __name__ == "__main__":
    main()
