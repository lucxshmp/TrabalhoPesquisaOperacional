import argparse
import os
import sys

BCP_DIR = os.path.join(os.path.dirname(__file__), "..", "BCP")
sys.path.insert(0, os.path.abspath(BCP_DIR))

import numpy as np
from instance_reader import load_instance
from solution import Solution
from construction import seed_routes, best_insertion_cimbm, best_insertion_cimp, GAMMA_VALUES

ILS_DIR = os.path.dirname(__file__)
DEFAULT_DADOS  = os.path.join(ILS_DIR, "datasets", "dadosPO.xlsx")
DEFAULT_OUTPUT = os.path.join(ILS_DIR, "resultados")


def parse_args():
    parser = argparse.ArgumentParser(
        description="ILS-RVND para o VRP (Kramer, Subramanian & Penna, 2016)"
    )
    parser.add_argument("--dados", default=DEFAULT_DADOS,
                        help="Planilha única com abas 'concessionárias' e 'pedidos'")
    parser.add_argument("--depot-lat", type=float, default=-19.9,
                        help="Latitude do depósito (default: -19.9)")
    parser.add_argument("--depot-lon", type=float, default=-44.1,
                        help="Longitude do depósito (default: -44.1)")
    parser.add_argument("--Q", type=int, default=100,
                        help="Capacidade do veículo (default: 100)")
    parser.add_argument("--Y", type=float, default=8.0,
                        help="Duração máxima da rota em horas (default: 8)")
    parser.add_argument("--K", type=int, default=20,
                        help="Tamanho da frota (default: 20)")
    parser.add_argument("--runs", type=int, default=10,
                        help="Número de execuções do ILS (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semente do RNG para reprodutibilidade (default: 42)")
    parser.add_argument("--maxiter", type=int, default=10,
                        help="MaxIter: reinícios do ILS (default: 10)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT,
                        help="Diretório para CSVs e gráficos gerados")
    parser.add_argument("--debug", action="store_true",
                        help="Modo debug: valida deltas incrementais a cada movimento")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=== ILS-RVND ===")
    print(f"Dados           : {args.dados}")
    print(f"Depósito        : lat={args.depot_lat}, lon={args.depot_lon}")
    print(f"Parâmetros VRP  : Q={args.Q}  Y={args.Y}h  K={args.K}")
    print(f"ILS             : MaxIter={args.maxiter}  runs={args.runs}  seed={args.seed}")
    print(f"Debug           : {args.debug}")
    print(f"Output          : {args.output_dir}")
    print()

    print("Carregando instância...")
    instance = load_instance(
        path=args.dados,
        depot_lat=args.depot_lat,
        depot_lon=args.depot_lon,
        Q=args.Q,
        Y=args.Y,
        K=args.K,
    )

    # converte matrizes para numpy
    instance.cost_matrix = np.array(instance.cost_matrix, dtype=float)
    instance.time_matrix = np.array(instance.time_matrix, dtype=float)

    n_clientes = len(instance.nodes) - 1
    total_demand = sum(nd.demanda for nd in instance.nodes[1:])
    min_routes = -(-int(total_demand) // instance.Q)  # ceil division
    print(f"\nClientes (nós)       : {n_clientes}")
    print(f"Demanda total        : {total_demand}")
    print(f"Mínimo teórico rotas : {min_routes}")
    print(f"Dimensão cost_matrix : {instance.cost_matrix.shape}")
    print(f"Dimensão time_matrix : {instance.time_matrix.shape}")

    # Smoke test: rotas ida-e-volta triviais (uma rota por cliente)
    print("\n--- Smoke test: rotas ida-e-volta ---")
    routes_trivial = [[i] for i in range(1, len(instance.nodes))]
    sol_trivial = Solution.from_routes(instance, routes_trivial)
    ok, errors = sol_trivial.validate()
    if ok:
        print(f"Solução trivial: custo={sol_trivial.cost:.4f} km  [VÁLIDA]")
    else:
        print("Solução trivial INVÁLIDA:")
        for e in errors:
            print(f"  {e}")

    # T2.1 — semeadura por maior demanda
    print("\n--- T2.1: semeadura por maior demanda ---")
    routes_seed, pending = seed_routes(instance)
    seeds = [r[0] for r in routes_seed]
    seed_demands = [instance.nodes[v].demanda for v in seeds]
    assert len(set(seeds)) == len(seeds), "clientes repetidos nas sementes!"
    assert all(v not in seeds for v in pending), "pendente aparece como semente!"
    print(f"Rotas semeadas   : {len(routes_seed)}  (K={instance.K})")
    print(f"Clientes pending : {len(pending)}")
    print(f"Demandas sementes: {seed_demands[:5]}{'...' if len(seed_demands) > 5 else ''}")
    print("Semeadura OK [DoD T2.1 verificado]")

    # T2.2 — critérios de inserção CIMBM e CIMP
    print("\n--- T2.2: critérios de inserção ---")
    c = instance.cost_matrix
    # usa a primeira rota semeada e o primeiro cliente pendente como exemplo
    rota_ex   = routes_seed[0]
    cliente_ex = pending[0]

    # CIMP: inserção mais próxima (delta real de custo)
    pos_cimp, delta_cimp = best_insertion_cimp(rota_ex, cliente_ex, c)
    # verifica manualmente: custo deve ser c[prev][k] + c[k][next] - c[prev][next]
    full = [0] + rota_ex + [0]
    i_ex, j_ex = full[pos_cimp], full[pos_cimp + 1]
    delta_manual = c[i_ex][cliente_ex] + c[cliente_ex][j_ex] - c[i_ex][j_ex]
    assert abs(delta_cimp - delta_manual) < 1e-9, "CIMP: delta incorreto!"

    # CIMBM: com γ = 0 deve igualar o CIMP (penalidade de depósito nula)
    pos_cimbm0, custo_cimbm0 = best_insertion_cimbm(rota_ex, cliente_ex, c, gamma=0.0)
    assert pos_cimbm0 == pos_cimp, "CIMBM(γ=0) deve escolher mesma posição que CIMP!"

    # CIMBM: com γ máximo (1.70)
    pos_cimbm_max, custo_cimbm_max = best_insertion_cimbm(rota_ex, cliente_ex, c, gamma=1.70)

    print(f"Rota exemplo     : {rota_ex}  →  cliente {cliente_ex}")
    print(f"CIMP: pos={pos_cimp}  delta={delta_cimp:.4f} km")
    print(f"CIMBM(γ=0.00): pos={pos_cimbm0}  custo={custo_cimbm0:.4f}")
    print(f"CIMBM(γ=1.70): pos={pos_cimbm_max}  custo={custo_cimbm_max:.4f}")
    print(f"γ disponíveis    : {len(GAMMA_VALUES)} valores ({GAMMA_VALUES[0]}..{GAMMA_VALUES[-1]})")
    print("Critérios OK [DoD T2.2 verificado]")


if __name__ == "__main__":
    main()
