"""
Experimentos do ILS-RVND (T5.1) — réplica da Tabela 3 do artigo.

Roda o ILS N vezes (default 10) por instância, cada execução com uma semente
distinta (base_seed + i, p/ reprodutibilidade), e agrega:
  - melhor custo, custo médio e desvio-padrão (amostral, ddof=1);
  - tempo médio (e total) de execução.

Gera em `ILS/resultados/`:
  - experimentos_resumo.csv     — uma linha por instância (a tabela do artigo);
  - experimentos_execucoes.csv  — uma linha por execução (alimenta o boxplot, T5.2);
  - convergencia.csv            — histórico (tempo, custo) por execução (curva, T5.2).

Uso como script:
    python experiments.py --runs 10 --max-clientes 300 --seed 42
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass

import numpy as np

from solution import Solution
from ils import ils


@dataclass
class RunResult:
    run: int
    seed: int
    cost: float
    time_s: float
    n_routes: int
    history: list  # lista de (tempo_s, custo_melhor_global)


# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------

def run_ils_experiment(instance, runs=10, base_seed=42, maxiter=10, maxiterils=None):
    """Roda o ILS `runs` vezes. Retorna (lista de RunResult, melhor Solution)."""
    results: list[RunResult] = []
    best_sol: Solution | None = None
    for i in range(runs):
        seed = base_seed + i
        rng = np.random.default_rng(seed)
        t0 = time.perf_counter()
        sol, history = ils(instance, rng, maxiter=maxiter, maxiterils=maxiterils)
        elapsed = time.perf_counter() - t0
        n_routes = sum(1 for r in sol.routes if r)
        results.append(RunResult(i, seed, sol.cost, elapsed, n_routes, history))
        if best_sol is None or sol.cost < best_sol.cost:
            best_sol = sol.copy()
    return results, best_sol


def summarize(instance_name, instance, results, maxiter, maxiterils):
    """Agrega os resultados em uma linha-resumo (réplica da Tabela 3)."""
    costs = np.array([r.cost for r in results], dtype=float)
    times = np.array([r.time_s for r in results], dtype=float)
    ddof = 1 if len(costs) > 1 else 0     # desvio-padrão amostral
    n_clientes = len(instance.nodes) - 1
    if maxiterils is None:
        maxiterils = n_clientes + instance.K
    return {
        "instancia": instance_name,
        "n_clientes": n_clientes,
        "runs": len(results),
        "maxiter": maxiter,
        "maxiterils": maxiterils,
        "melhor_custo": round(float(costs.min()), 4),
        "media_custo": round(float(costs.mean()), 4),
        "desvio_custo": round(float(costs.std(ddof=ddof)), 4),
        "tempo_medio_s": round(float(times.mean()), 2),
        "tempo_total_s": round(float(times.sum()), 2),
    }


# ---------------------------------------------------------------------------
# Escrita dos CSVs
# ---------------------------------------------------------------------------

SUMMARY_COLUMNS = ["instancia", "n_clientes", "runs", "maxiter", "maxiterils",
                   "melhor_custo", "media_custo", "desvio_custo",
                   "tempo_medio_s", "tempo_total_s"]
RUNS_COLUMNS = ["instancia", "run", "seed", "custo", "tempo_s", "n_rotas"]
CONV_COLUMNS = ["instancia", "run", "tempo_s", "custo"]


def _write_csv(path, columns, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path, summary_rows):
    _write_csv(path, SUMMARY_COLUMNS, summary_rows)


def write_runs_csv(path, instance_name, results):
    rows = [{"instancia": instance_name, "run": r.run, "seed": r.seed,
             "custo": round(r.cost, 4), "tempo_s": round(r.time_s, 2),
             "n_rotas": r.n_routes}
            for r in results]
    _write_csv(path, RUNS_COLUMNS, rows)


def write_convergence_csv(path, instance_name, results):
    rows = [{"instancia": instance_name, "run": r.run,
             "tempo_s": round(t, 4), "custo": round(c, 4)}
            for r in results for (t, c) in r.history]
    _write_csv(path, CONV_COLUMNS, rows)


def run_and_save(instance, instance_name, output_dir, runs=10, base_seed=42,
                 maxiter=10, maxiterils=None):
    """Roda o experimento e grava os três CSVs. Retorna (resumo, results, melhor)."""
    results, best = run_ils_experiment(instance, runs, base_seed, maxiter, maxiterils)
    summary = summarize(instance_name, instance, results, maxiter, maxiterils)
    write_summary_csv(os.path.join(output_dir, "experimentos_resumo.csv"), [summary])
    write_runs_csv(os.path.join(output_dir, "experimentos_execucoes.csv"),
                   instance_name, results)
    write_convergence_csv(os.path.join(output_dir, "convergencia.csv"),
                          instance_name, results)
    return summary, results, best


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Experimentos do ILS-RVND (T5.1)")
    ils_dir = os.path.dirname(__file__)
    p.add_argument("--dados", default=os.path.join(ils_dir, "datasets", "dadosPO.xlsx"))
    p.add_argument("--instancia", default="dadosPO-300",
                   help="Rótulo da instância no CSV")
    p.add_argument("--depot-lat", type=float, default=-19.9)
    p.add_argument("--depot-lon", type=float, default=-44.1)
    p.add_argument("--Q", type=int, default=100)
    p.add_argument("--Y", type=float, default=8.0)
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--max-clientes", type=int, default=300,
                   help="Limita aos primeiros N clientes (0 = todos)")
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--maxiter", type=int, default=10)
    p.add_argument("--maxiterils", type=int, default=None)
    p.add_argument("--output-dir", default=os.path.join(ils_dir, "resultados"))
    return p.parse_args()


def main():
    from instance_reader import load_instance
    args = _parse_args()

    print(f"Carregando instância (max_clientes={args.max_clientes or 'todos'})...")
    instance = load_instance(
        path=args.dados, depot_lat=args.depot_lat, depot_lon=args.depot_lon,
        Q=args.Q, Y=args.Y, K=args.K, max_clientes=(args.max_clientes or None),
    )
    instance.cost_matrix = np.array(instance.cost_matrix, dtype=float)
    instance.time_matrix = np.array(instance.time_matrix, dtype=float)

    n = len(instance.nodes) - 1
    print(f"Clientes: {n}  | runs={args.runs}  seed={args.seed}  "
          f"maxiter={args.maxiter}  maxiterils={args.maxiterils or (n + args.K)}")
    print("Rodando ILS...")

    summary, _, best = run_and_save(
        instance, args.instancia, args.output_dir,
        runs=args.runs, base_seed=args.seed,
        maxiter=args.maxiter, maxiterils=args.maxiterils,
    )

    print("\n=== Resumo (réplica da Tabela 3) ===")
    print(f"  Instância      : {summary['instancia']}  ({summary['n_clientes']} clientes)")
    print(f"  Execuções      : {summary['runs']}")
    print(f"  Melhor custo   : {summary['melhor_custo']} km")
    print(f"  Custo médio    : {summary['media_custo']} km")
    print(f"  Desvio-padrão  : {summary['desvio_custo']} km")
    print(f"  Tempo médio    : {summary['tempo_medio_s']} s")
    print(f"\nCSVs em: {args.output_dir}")
    ok, errs = best.validate()
    print(f"Melhor solução válida: {ok}" + ("" if ok else f"  ({errs[0]})"))


if __name__ == "__main__":
    main()
