import argparse
import os
import sys

BCP_DIR = os.path.join(os.path.dirname(__file__), "..", "BCP")
sys.path.insert(0, os.path.abspath(BCP_DIR))

import numpy as np
from instance_reader import load_instance
from solution import Solution
from construction import (
    seed_routes, best_insertion_cimbm, best_insertion_cimp,
    GAMMA_VALUES, build_solution,
)

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
    parser.add_argument("--max-clientes", type=int, default=300,
                        help="Limita aos primeiros N clientes (default: 300; use 0 p/ todos)")
    parser.add_argument("--runs", type=int, default=10,
                        help="Número de execuções do ILS (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semente do RNG para reprodutibilidade (default: 42)")
    parser.add_argument("--maxiter", type=int, default=10,
                        help="MaxIter: reinícios do ILS (default: 10)")
    parser.add_argument("--maxiterils", type=int, default=None,
                        help="MaxIterILS: iterações do laço interno (default: n + K)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT,
                        help="Diretório para CSVs e gráficos gerados")
    parser.add_argument("--debug", action="store_true",
                        help="Modo debug: valida deltas incrementais a cada movimento")
    parser.add_argument("--verify", default="all",
                        help="Blocos de verificação a rodar: 'all' ou lista separada "
                             "por vírgula (t12,t31,t32,t33,t41,t42,t43). Ex.: --verify t43")
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
        max_clientes=(args.max_clientes or None),  # 0 → None (todos)
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

    run_verifications(instance, args)


def verify_t12(instance, args):
    """T1.3–T2.x — baseline, semeadura, critérios, estratégias e veículo extra."""
    # Smoke test / T1.3 — baseline ida-e-volta (uma rota por cliente)
    print("\n--- Baseline ida-e-volta (T1.3) ---")
    routes_trivial = [[i] for i in range(1, len(instance.nodes))]
    sol_trivial = Solution.from_routes(instance, routes_trivial)
    baseline_cost = sol_trivial.cost
    print(f"Custo baseline   : {baseline_cost:.1f} km  (viola frota — apenas referência de custo)")

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

    # T2.3 / T2.5 — estratégias EIS e EIP + comparação com baseline
    print("\n--- T2.3 / T2.5: construção vs. baseline ---")

    for label, seed_offset in [("EIS", 0), ("EIP", 1)]:
        rng_test = np.random.default_rng(args.seed + seed_offset)
        routes_built, leftover = build_solution(instance, rng_test)
        sol = Solution.from_routes(instance, routes_built)
        ok, errors = sol.validate()
        status = "VÁLIDA" if ok else f"inválida ({errors[0]})"
        reducao = (baseline_cost - sol.cost) / baseline_cost * 100
        assert sol.cost <= baseline_cost, f"{label}: custo ({sol.cost:.1f}) > baseline ({baseline_cost:.1f})!"
        print(
            f"  {label}: custo={sol.cost:.1f} km  "
            f"redução={reducao:.1f}% vs baseline  [{status}]"
        )

    print("T2.3 / T2.5 verificados: construção < baseline")

    # T2.4 — veículo extra: força instância apertada com K=2 e Y pequeno
    print("\n--- T2.4: veículo extra (instância apertada K=2, Y=0.1h) ---")
    from data.structures import Instance as _Instance
    inst_apertada = _Instance(
        nodes=instance.nodes,
        cost_matrix=instance.cost_matrix,
        time_matrix=instance.time_matrix,
        Q=instance.Q,
        Y=0.1,   # duração máxima muito pequena → quase ninguém cabe em rota alheia
        K=2,
    )
    rng_t24 = np.random.default_rng(args.seed)
    routes_t24, leftover_t24 = build_solution(inst_apertada, rng_t24)
    n_extra = max(0, len([r for r in routes_t24 if r]) - inst_apertada.K)
    assert leftover_t24 == [], "pending deveria estar vazio após T2.4!"
    sol_t24 = Solution.from_routes(inst_apertada, routes_t24)
    ok_t24, errs_t24 = sol_t24.validate()
    print(f"  Rotas totais     : {len([r for r in routes_t24 if r])}")
    print(f"  Veículos extras  : {n_extra}")
    print(f"  Clientes pending : {len(leftover_t24)}")
    print(f"  Validação        : {'OK' if ok_t24 else errs_t24[0]}")
    assert not ok_t24, "solução com veículo extra deve ser sinalizada como inválida!"
    print("T2.4 verificado: instância apertada não travou; extras sinalizados")


# Registro ordenado dos blocos de verificação (--verify seleciona quais rodar)
def _verifications():
    return {
        "t12": verify_t12,
        "t31": verify_t31,
        "t32": verify_t32,
        "t33": verify_t33,
        "t41": verify_t41,
        "t42": verify_t42,
        "t43": verify_t43,
        "t44": verify_t44,
        "t51": verify_t51,
    }


def run_verifications(instance, args):
    """Roda os blocos pedidos em --verify (default: todos, na ordem)."""
    blocos = _verifications()
    escolha = (args.verify or "all").strip().lower()
    if escolha in ("all", "todos", ""):
        selecionados = list(blocos)
    else:
        selecionados = [k.strip() for k in escolha.split(",") if k.strip()]
        desconhecidos = [k for k in selecionados if k not in blocos]
        if desconhecidos:
            raise SystemExit(
                f"--verify inválido: {desconhecidos}. "
                f"Opções: {', '.join(blocos)} ou 'all'."
            )
    print(f"\n[verify] blocos selecionados: {', '.join(selecionados)}")
    for k in selecionados:
        blocos[k](instance, args)


def _apply_until_stable(viz, sol, nome):
    """Aplica `viz` (best-improvement) em laço até não melhorar, com asserts."""
    n_mov = 0
    while True:
        custo_antes = sol.cost
        if not viz(sol):
            break
        n_mov += 1
        assert sol.cost <= custo_antes + 1e-9, \
            f"{nome}: custo aumentou ({custo_antes:.4f} -> {sol.cost:.4f})!"
        ok_v, errs_v = sol.validate()
        assert ok_v, f"{nome}: validate falhou após movimento: {errs_v[0]}"
    return n_mov


def verify_t31(instance, args):
    """T3.1 — vizinhanças intrarrotas (Reinserção, Or-opt2, Or-opt3, Exchange)."""
    print("\n--- T3.1: vizinhanças intrarrotas ---")
    from neighborhoods import reinsertion, or_opt2, or_opt3, exchange

    rng = np.random.default_rng(args.seed)
    routes, _ = build_solution(instance, rng)
    # embaralha a ordem interna de cada rota para torná-la subótima e, assim,
    # exercitar de fato as vizinhanças (a construção já fica perto do ótimo intra)
    for r in routes:
        rng.shuffle(r)
    sol = Solution.from_routes(instance, routes)
    custo_inicial = sol.cost

    for nome, viz in [("Reinserção", reinsertion), ("Or-opt2", or_opt2),
                      ("Or-opt3", or_opt3), ("Exchange", exchange)]:
        n_mov = _apply_until_stable(viz, sol, nome)
        print(f"  {nome:11s}: {n_mov} movimento(s) aplicado(s)  custo={sol.cost:.1f} km")

    reducao = (custo_inicial - sol.cost) / custo_inicial * 100
    print(f"  Custo: {custo_inicial:.1f} -> {sol.cost:.1f} km  (-{reducao:.1f}% via intrarrotas)")
    ok_final, _ = sol.validate()
    assert ok_final, "solução final das intrarrotas deve ser válida!"
    print("T3.1 verificado: cada vizinhança não piora o custo; validate passa")


def verify_t32(instance, args):
    """T3.2 — vizinhanças inter-rotas (Shift/Swap/Cross)."""
    print("\n--- T3.2: vizinhanças inter-rotas ---")
    from neighborhoods import shift10, shift20, swap11, swap21, swap22, cross

    rng = np.random.default_rng(args.seed)
    routes, _ = build_solution(instance, rng)
    sol_base = Solution.from_routes(instance, routes)

    # cada vizinhança parte da MESMA solução construída → exercita seu delta
    for nome, viz in [("Shift(1,0)", shift10), ("Shift(2,0)", shift20),
                      ("Swap(1,1)", swap11), ("Swap(2,1)", swap21),
                      ("Swap(2,2)", swap22), ("Cross", cross)]:
        sol = sol_base.copy()
        n_mov = _apply_until_stable(viz, sol, nome)
        red = (sol_base.cost - sol.cost) / sol_base.cost * 100
        print(f"  {nome:11s}: {n_mov:3d} mov.  {sol_base.cost:.1f} -> {sol.cost:.1f} km  (-{red:.1f}%)")

    print("T3.2 verificado: deltas corretos; cargas/durações das 2 rotas atualizadas")


def verify_t33(instance, args):
    """T3.3 — modo debug de deltas: suíte de movimentos aleatórios com asserts."""
    print("\n--- T3.3: modo debug de deltas (suíte aleatória) ---")
    import neighborhoods
    from neighborhoods import INTRA_NEIGHBORHOODS, INTER_NEIGHBORHOODS

    todas_viz = INTRA_NEIGHBORHOODS + INTER_NEIGHBORHOODS
    rng = np.random.default_rng(args.seed + 7)
    routes, _ = build_solution(instance, rng)
    for r in routes:                # embaralha p/ gerar oportunidades de melhora
        rng.shuffle(r)
    sol = Solution.from_routes(instance, routes)
    custo_inicial = sol.cost

    neighborhoods.set_debug(True)   # liga a recomputação+assert a cada movimento
    n_aplicados = 0
    try:
        sem_melhora = 0
        # sorteia vizinhanças até estagnar (nenhuma das 10 melhora em sequência)
        while sem_melhora < len(todas_viz):
            viz = todas_viz[rng.integers(0, len(todas_viz))]
            if viz(sol):            # se aplicou, o decorator já fez assert_consistent
                n_aplicados += 1
                sem_melhora = 0
            else:
                sem_melhora += 1
    finally:
        neighborhoods.set_debug(bool(args.debug))  # restaura conforme a flag

    ok, errs = sol.validate()
    assert ok, f"solução final da suíte debug inválida: {errs[0]}"
    red = (custo_inicial - sol.cost) / custo_inicial * 100
    print(f"  Movimentos aplicados (todos checados): {n_aplicados}")
    print(f"  Custo: {custo_inicial:.1f} -> {sol.cost:.1f} km  (-{red:.1f}%)")
    print("T3.3 verificado: suíte aleatória rodou com debug sem disparar assert")


def verify_t41(instance, args):
    """T4.1 — RVND: converge a ótimo local; custo nunca piora; validate passa."""
    print("\n--- T4.1: RVND (Algoritmo 4) ---")
    import neighborhoods
    from neighborhoods import INTRA_NEIGHBORHOODS, INTER_NEIGHBORHOODS
    from rvnd import rvnd

    rng = np.random.default_rng(args.seed)
    routes, _ = build_solution(instance, rng)
    sol = Solution.from_routes(instance, routes)
    custo_inicial = sol.cost

    # debug ligado: garante que o custo nunca piora e os deltas batem a cada passo
    neighborhoods.set_debug(True)
    try:
        rvnd(sol, rng)
    finally:
        neighborhoods.set_debug(bool(args.debug))

    assert sol.cost <= custo_inicial + 1e-9, \
        f"RVND piorou o custo ({custo_inicial:.1f} -> {sol.cost:.1f})!"
    ok, errs = sol.validate()
    assert ok, f"solução do RVND inválida: {errs[0]}"

    # ótimo local: nenhuma das 10 vizinhanças encontra melhora adicional
    todas_viz = INTER_NEIGHBORHOODS + INTRA_NEIGHBORHOODS
    assert not any(viz(sol.copy()) for viz in todas_viz), \
        "RVND não convergiu: ainda há vizinhança com movimento de melhora!"

    red = (custo_inicial - sol.cost) / custo_inicial * 100
    n_rotas = sum(1 for r in sol.routes if r)
    print(f"  Custo: {custo_inicial:.1f} -> {sol.cost:.1f} km  (-{red:.1f}%)")
    print(f"  Rotas não-vazias: {n_rotas}/{instance.K}")
    print("T4.1 verificado: RVND em ótimo local; custo não piorou; validate passa")


def verify_t42(instance, args):
    """T4.2 — perturbações: alteram a solução mantendo viabilidade."""
    print("\n--- T4.2: perturbações (Multiple-Swap / Multiple-Shift) ---")
    from perturbation import multiple_swap, multiple_shift, perturb
    from rvnd import rvnd

    rng = np.random.default_rng(args.seed)
    routes, _ = build_solution(instance, rng)
    sol_base = Solution.from_routes(instance, routes)

    # cada perturbação isoladamente: altera a solução e mantém viabilidade
    for nome, pert in [("Multiple-Swap", multiple_swap), ("Multiple-Shift", multiple_shift)]:
        sol = sol_base.copy()
        antes = sol.cost
        n = pert(sol, rng, n_moves=10)
        sol.assert_consistent(context=nome)       # deltas batem com recomputação
        ok, errs = sol.validate()                  # viabilidade (Q, Y, cobertura)
        assert ok, f"{nome}: solução perturbada inválida: {errs[0]}"
        assert n > 0 and abs(sol.cost - antes) > 1e-9, \
            f"{nome}: perturbação não alterou a solução!"
        print(f"  {nome:14s}: {n:2d} mov.  custo {antes:.1f} -> {sol.cost:.1f} km  [viável]")

    # perturbação sorteada repetida, sempre checando consistência/viabilidade
    sol = sol_base.copy()
    for _ in range(20):
        perturb(sol, rng)
        sol.assert_consistent(context="perturb")
        ok, errs = sol.validate()
        assert ok, f"perturb: solução inválida: {errs[0]}"

    # reparabilidade: após perturbar, o RVND volta a um ótimo local viável
    sol = sol_base.copy()
    custo_opt = rvnd(sol.copy(), np.random.default_rng(args.seed)).cost
    perturb(sol, rng, n_moves=15)
    rvnd(sol, rng)
    ok, errs = sol.validate()
    assert ok, f"reparabilidade: solução pós-RVND inválida: {errs[0]}"
    print(f"  Reparabilidade : perturba e RVND repara → custo={sol.cost:.1f} km "
          f"(ótimo local direto={custo_opt:.1f}) [viável]")
    print("T4.2 verificado: perturbações alteram a solução mantendo viabilidade")


def verify_t43(instance, args):
    """T4.3 — ILS: retorna s* viável, melhor que a construção, com histórico."""
    print("\n--- T4.3: ILS (Algoritmo 2) ---")
    from ils import ils

    # custo de referência: melhor construção pura entre algumas tentativas
    rng_ref = np.random.default_rng(args.seed)
    custo_constr = min(
        Solution.from_routes(instance, build_solution(instance, rng_ref)[0]).cost
        for _ in range(3)
    )

    # com 300 clientes o ILS roda em ~1 min com params suficientes p/ uma
    # curva de convergência de verdade (runs do T5 usam --maxiter / n+K)
    maxiter = max(1, min(args.maxiter, 3))
    maxiterils = args.maxiterils if args.maxiterils is not None else 20
    rng = np.random.default_rng(args.seed)
    print(f"  Config smoke    : maxiter={maxiter}  maxiterils={maxiterils}  seed={args.seed}")
    best, history = ils(instance, rng, maxiter=maxiter, maxiterils=maxiterils)

    ok, errs = best.validate()
    assert ok, f"s* do ILS inválida: {errs[0]}"
    assert best.cost < custo_constr - 1e-9, \
        f"ILS ({best.cost:.1f}) não melhorou sobre a construção ({custo_constr:.1f})!"

    # histórico populado e monotonicamente não-crescente (melhor global)
    assert len(history) > 0, "histórico de convergência vazio!"
    custos = [c for _, c in history]
    tempos = [tm for tm, _ in history]
    assert all(custos[i] >= custos[i + 1] - 1e-9 for i in range(len(custos) - 1)), \
        "custo da melhor global aumentou no histórico!"
    assert all(tempos[i] <= tempos[i + 1] + 1e-9 for i in range(len(tempos) - 1)), \
        "tempos do histórico fora de ordem!"

    n_rotas = sum(1 for r in best.routes if r)
    red = (custo_constr - best.cost) / custo_constr * 100
    print(f"  Construção pura : {custo_constr:.1f} km")
    print(f"  ILS (s*)        : {best.cost:.1f} km  (-{red:.1f}%)  rotas {n_rotas}/{instance.K}")
    print(f"  Histórico       : {len(history)} pontos  "
          f"({history[0][1]:.1f} -> {history[-1][1]:.1f} km em {history[-1][0]:.1f}s)")
    print("T4.3 verificado: s* viável, melhora sobre a construção, histórico populado")


def verify_t44(instance, args):
    """T4.4 — determinismo: mesma --seed → custo (e rotas) idênticos."""
    print("\n--- T4.4: determinismo ---")
    from ils import ils

    def run():
        # RNG único, derivado só de --seed, propagado a construção/RVND/perturbação
        rng = np.random.default_rng(args.seed)
        best, _ = ils(instance, rng, maxiter=1, maxiterils=5)
        return best

    b1 = run()
    b2 = run()

    assert b1.cost == b2.cost, \
        f"determinismo falhou: custos diferentes ({b1.cost!r} != {b2.cost!r})"
    assert b1.routes == b2.routes, \
        "determinismo falhou: mesmas seeds produziram rotas diferentes!"

    ok, errs = b1.validate()
    assert ok, f"s* inválida no teste de determinismo: {errs[0]}"
    print(f"  Execução 1 (seed={args.seed}): {b1.cost:.6f} km")
    print(f"  Execução 2 (seed={args.seed}): {b2.cost:.6f} km")
    print("T4.4 verificado: mesma seed → custo e rotas idênticos")


def verify_t51(instance, args):
    """T5.1 — experiments.py: gera CSV-resumo com melhor/média/desvio/tempo."""
    print("\n--- T5.1: experiments.py (CSV de resultados) ---")
    import csv
    import os
    from experiments import run_and_save

    # config reduzida p/ o smoke test (runs reais: python experiments.py --runs 10)
    summary, results, best = run_and_save(
        instance, "smoke-300", args.output_dir,
        runs=3, base_seed=args.seed, maxiter=1, maxiterils=3,
    )

    path = os.path.join(args.output_dir, "experimentos_resumo.csv")
    assert os.path.exists(path), f"CSV não gerado: {path}"
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        rows = list(reader)

    for needed in ["melhor_custo", "media_custo", "desvio_custo", "tempo_medio_s"]:
        assert needed in cols, f"coluna obrigatória ausente no CSV: {needed}"
    assert len(rows) >= 1, "CSV-resumo sem linhas de dados!"
    assert len(results) == 3, "número de execuções inesperado!"
    ok, errs = best.validate()
    assert ok, f"melhor solução do experimento inválida: {errs[0]}"

    print(f"  CSV resumo     : {path}")
    print(f"  Colunas        : {cols}")
    print(f"  melhor={summary['melhor_custo']}  média={summary['media_custo']}  "
          f"desvio={summary['desvio_custo']}  tempo_médio={summary['tempo_medio_s']}s")
    print("T5.1 verificado: CSV gerado em resultados/ com melhor/média/desvio/tempo")


if __name__ == "__main__":
    main()
