"""
Cenários de execução do ILS-RVND.

Roda a metaheurística em 3 cenários (frota × período dos pedidos) e salva, para
cada um, os CSVs (experiments.py) e os 3 gráficos (plots.py) numa subpasta
própria em `resultados/cenarios/`:

  1) 15 caminhões — pedidos de um dia      (2026-06-05)
  2) 10 caminhões — pedidos de uma semana  (2026-06-01 a 06-07)
  3) 10 caminhões — pedidos de um mês       (junho/2026 inteiro)

Observações sobre os dados:
  - 1 nó por pedido (modelo do projeto); ~10 locais distintos geocodificáveis.
  - demanda = 1 por pedido; Q=100, Y=8h (padrões do projeto).
  - tamanhos (após geocodificação): dia≈165, semana≈425, mês≈669 nós.
  - como é Python puro, os params do ILS são escalados por tamanho p/ caber em
    tempo razoável (use --rapido p/ um teste ainda mais curto).

Uso:
    python scenarios.py            # roda os 3 cenários
    python scenarios.py --rapido   # params mínimos (validação rápida)
"""

from __future__ import annotations

import os
import time

import numpy as np

from instance_reader import load_instance
from experiments import run_and_save
from plots import (read_convergence, read_runs, read_routes,
                   plot_convergence, plot_boxplot, plot_routes)


# Cada cenário: rótulo, frota K, janela de datas e params do ILS (escala c/ tamanho)
SCENARIOS = [
    {
        "nome": "cenario1_dia", "titulo": "15 caminhões — 1 dia (2026-06-05)",
        "K": 15, "data_inicio": "2026-06-05", "data_fim": "2026-06-05",
        "runs": 5, "maxiter": 2, "maxiterils": 12,
    },
    {
        "nome": "cenario2_semana", "titulo": "10 caminhões — 1 semana (06-01 a 06-07)",
        "K": 10, "data_inicio": "2026-06-01", "data_fim": "2026-06-07",
        "runs": 3, "maxiter": 2, "maxiterils": 6,
    },
    {
        "nome": "cenario3_mes", "titulo": "10 caminhões — 1 mês (junho/2026)",
        "K": 10, "data_inicio": "2026-06-01", "data_fim": "2026-06-30",
        "runs": 3, "maxiter": 1, "maxiterils": 20,
    },
]


def run_scenario(sc, base_dir, dados, depot_lat, depot_lon, Q, Y, seed, rapido=False):
    out = os.path.join(base_dir, sc["nome"])
    print(f"\n========== {sc['titulo']} ==========")

    instance = load_instance(
        path=dados, depot_lat=depot_lat, depot_lon=depot_lon,
        Q=Q, Y=Y, K=sc["K"],
        data_inicio=sc["data_inicio"], data_fim=sc["data_fim"],
    )
    instance.cost_matrix = np.array(instance.cost_matrix, dtype=float)
    instance.time_matrix = np.array(instance.time_matrix, dtype=float)
    n = len(instance.nodes) - 1

    runs = 2 if rapido else sc["runs"]
    maxiter = 1 if rapido else sc["maxiter"]
    maxiterils = 3 if rapido else sc["maxiterils"]
    print(f"  nós={n}  K={sc['K']}  Q={Q}  Y={Y}h  | "
          f"runs={runs} maxiter={maxiter} maxiterils={maxiterils}")

    t0 = time.perf_counter()
    summary, _results, best = run_and_save(
        instance, sc["nome"], out,
        runs=runs, base_seed=seed, maxiter=maxiter, maxiterils=maxiterils,
    )
    elapsed = time.perf_counter() - t0

    # gráficos a partir dos CSVs recém-gravados
    plot_convergence(read_convergence(os.path.join(out, "convergencia.csv")), out)
    plot_boxplot(read_runs(os.path.join(out, "experimentos_execucoes.csv")), out)
    rotas = read_routes(os.path.join(out, "melhor_rotas.csv"))
    for inst, routes in rotas.items():
        plot_routes(instance, routes, out, inst)

    ok, errs = best.validate()
    n_rotas = sum(1 for r in best.routes if r)
    print(f"  → melhor={summary['melhor_custo']} km  média={summary['media_custo']} "
          f"desvio={summary['desvio_custo']}  rotas={n_rotas}/{sc['K']}  "
          f"válida={ok}{'' if ok else ' (' + errs[0] + ')'}")
    print(f"  → tempo total cenário: {elapsed:.0f}s  | CSVs+PNGs em: {out}")
    return {"cenario": sc["nome"], "titulo": sc["titulo"], "nos": n, "K": sc["K"],
            **{k: summary[k] for k in ("melhor_custo", "media_custo",
                                       "desvio_custo", "tempo_medio_s")},
            "rotas": n_rotas, "valida": ok}


def _parse_args():
    import argparse
    ils_dir = os.path.dirname(__file__)
    p = argparse.ArgumentParser(description="Cenários do ILS-RVND")
    p.add_argument("--dados", default=os.path.join(ils_dir, "datasets", "dadosPO.xlsx"))
    p.add_argument("--depot-lat", type=float, default=-19.9)
    p.add_argument("--depot-lon", type=float, default=-44.1)
    p.add_argument("--Q", type=int, default=100)
    p.add_argument("--Y", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir",
                   default=os.path.join(ils_dir, "resultados", "cenarios"))
    p.add_argument("--rapido", action="store_true",
                   help="params mínimos p/ validação rápida")
    return p.parse_args()


def main():
    args = _parse_args()
    print("=== Cenários ILS-RVND ===")
    resumos = []
    for sc in SCENARIOS:
        resumos.append(run_scenario(
            sc, args.output_dir, args.dados, args.depot_lat, args.depot_lon,
            args.Q, args.Y, args.seed, rapido=args.rapido,
        ))

    print("\n========== RESUMO DOS CENÁRIOS ==========")
    print(f"{'cenário':16s} {'nós':>4s} {'K':>3s} {'melhor':>9s} "
          f"{'média':>9s} {'desvio':>7s} {'rotas':>6s} {'tempo/run':>9s}")
    for r in resumos:
        print(f"{r['cenario']:16s} {r['nos']:4d} {r['K']:3d} "
              f"{r['melhor_custo']:9.1f} {r['media_custo']:9.1f} "
              f"{r['desvio_custo']:7.1f} {r['rotas']:6d} {r['tempo_medio_s']:8.1f}s")
    print(f"\nGráficos e CSVs por cenário em: {args.output_dir}")


if __name__ == "__main__":
    main()
