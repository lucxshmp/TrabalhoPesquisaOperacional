"""
Algoritmo 3 do artigo: construção de solução inicial para o ILS-RVND.

Fase de implementação:
  T2.1 - seed_routes: semeadura por maior demanda          ← implementado
  T2.2 - best_insertion_*: critérios CIMBM e CIMP         ← implementado
  T2.3 - build_solution: estratégias EIS e EIP            ← implementado
  T2.4 - regra do veículo extra                           ← implementado
"""

from __future__ import annotations
import numpy as np
from data.structures import Instance

# γ ∈ {0.00, 0.05, 0.10, ..., 1.70}  (35 valores, conforme o artigo)
GAMMA_VALUES: list[float] = [round(i * 0.05, 2) for i in range(35)]


def seed_routes(instance: Instance) -> tuple[list[list[int]], list[int]]:
    """
    Fase de semeadura (T2.1 — Algoritmo 3, passo inicial).

    Cria até K rotas, cada uma iniciada com o cliente de maior demanda
    ainda não atribuído. Os clientes restantes ficam na lista `pending`.

    Retorna:
        routes  — lista de K rotas (cada rota: lista com 1 nó semente).
        pending — clientes ainda não inseridos, em ordem decrescente de demanda.
    """
    K = instance.K
    customers = sorted(
        range(1, len(instance.nodes)),
        key=lambda v: instance.nodes[v].demanda,
        reverse=True,
    )

    n_seeds = min(K, len(customers))
    routes: list[list[int]] = [[customers[i]] for i in range(n_seeds)]
    pending: list[int] = list(customers[n_seeds:])

    return routes, pending


# ---------------------------------------------------------------------------
# T2.2 — Critérios de inserção
# ---------------------------------------------------------------------------

def best_insertion_cimbm(
    route: list[int],
    customer: int,
    cost_matrix: np.ndarray,
    gamma: float,
) -> tuple[int, float]:
    """
    CIMBM — Critério de Inserção Mais Barata Modificada (Algoritmo 3).

    Para cada posição p na rota (entre os nós i e j):
        custo(p) = c[i][k] + c[k][j] − c[i][j] − γ·(c[0][k] + c[k][0])

    O termo −γ·(...) favorece clientes distantes do depósito quando γ > 0,
    encorajando sua inserção nas rotas antes que fiquem isolados.

    Retorna (melhor_posição, custo_CIMBM_nessa_posição).
    Posição 0 = antes do primeiro nó da rota; posição len(route) = após o último.
    """
    c = cost_matrix
    k = customer
    depot_penalty = gamma * (c[0][k] + c[k][0])

    full = [0] + route + [0]  # depot → r[0] → ... → r[-1] → depot
    best_pos, best_cost = 0, float("inf")
    for pos in range(len(route) + 1):
        i, j = full[pos], full[pos + 1]
        cost = float(c[i][k] + c[k][j] - c[i][j]) - depot_penalty
        if cost < best_cost:
            best_cost = cost
            best_pos = pos

    return best_pos, best_cost


def best_insertion_cimp(
    route: list[int],
    customer: int,
    cost_matrix: np.ndarray,
) -> tuple[int, float]:
    """
    CIMP — Critério de Inserção Mais Próxima.

    Para cada posição p na rota (entre os nós i e j):
        delta(p) = c[i][k] + c[k][j] − c[i][j]

    Retorna (melhor_posição, delta_de_custo_nessa_posição).
    """
    c = cost_matrix
    k = customer

    full = [0] + route + [0]
    best_pos, best_cost = 0, float("inf")
    for pos in range(len(route) + 1):
        i, j = full[pos], full[pos + 1]
        cost = float(c[i][k] + c[k][j] - c[i][j])
        if cost < best_cost:
            best_cost = cost
            best_pos = pos

    return best_pos, best_cost


# ---------------------------------------------------------------------------
# T2.3 — Estratégias EIS e EIP
# ---------------------------------------------------------------------------

def _best_feasible_pos(
    route: list[int],
    customer: int,
    load_r: float,
    dur_r: float,
    instance: Instance,
    pos_cost_fn,
) -> tuple[int | None, float]:
    """
    Varre todas as posições de inserção e retorna a de menor custo (critério)
    que ainda respeite Q e Y. Retorna (None, inf) se nenhuma for viável.

    pos_cost_fn(i, k, j) → float: custo do critério para inserir k entre i e j.
    """
    t = instance.time_matrix
    new_load = load_r + instance.nodes[customer].demanda
    if new_load > instance.Q:
        return None, float("inf")

    full = [0] + route + [0]
    best_pos: int | None = None
    best_cost = float("inf")
    for pos in range(len(route) + 1):
        i, j = full[pos], full[pos + 1]
        dt = float(t[i][customer] + t[customer][j] - t[i][j])
        if dur_r + dt > instance.Y:
            continue
        cost = pos_cost_fn(i, customer, j)
        if cost < best_cost:
            best_cost = cost
            best_pos = pos
    return best_pos, best_cost


def _apply_insert(
    r_idx: int,
    customer: int,
    pos: int,
    routes: list[list[int]],
    loads: list[float],
    durations: list[float],
    instance: Instance,
) -> None:
    """Insere customer na posição pos da rota r_idx, atualizando acumuladores."""
    t = instance.time_matrix
    route = routes[r_idx]
    full = [0] + route + [0]
    i, j = full[pos], full[pos + 1]
    dt = float(t[i][customer] + t[customer][j] - t[i][j])
    route.insert(pos, customer)
    loads[r_idx] += instance.nodes[customer].demanda
    durations[r_idx] += dt


# ---------------------------------------------------------------------------
# T2.4 — Veículo extra
# ---------------------------------------------------------------------------

def _add_extra_vehicle(
    routes: list[list[int]],
    loads: list[float],
    durations: list[float],
    remaining: list[int],
    instance: Instance,
) -> None:
    """
    Adiciona uma rota extra (além das K originais) semeada com o cliente de
    maior demanda ainda pendente, removendo-o de remaining.

    Como a rota extra está além do índice K-1, Solution.validate() a detectará
    como violação de frota ("rotas não-vazias > K"), sinalizando a solução como
    inválida. O ILS descarta ou penaliza soluções com veículo extra.
    """
    seed = max(remaining, key=lambda v: instance.nodes[v].demanda)
    remaining.remove(seed)
    t = instance.time_matrix
    routes.append([seed])
    loads.append(float(instance.nodes[seed].demanda))
    durations.append(float(t[0][seed] + t[seed][0]))


def _build_eis(
    routes: list[list[int]],
    pending: list[int],
    loads: list[float],
    durations: list[float],
    instance: Instance,
    criterion_fn,
) -> list[int]:
    """
    EIS — Estratégia de Inserção Sequencial: preenche uma rota por vez.
    A cada passo, insere o cliente com menor custo de critério que seja viável.
    Quando todas as rotas atuais estiverem cheias, adiciona veículo extra (T2.4).
    """
    remaining = list(pending)
    r_idx = 0
    while remaining:
        if r_idx >= len(routes):
            # todas as rotas esgotadas → abre veículo extra (T2.4)
            _add_extra_vehicle(routes, loads, durations, remaining, instance)

        improved = True
        while improved and remaining:
            improved = False
            best_customer, best_pos, best_cost = None, None, float("inf")
            for customer in remaining:
                pos, cost = criterion_fn(routes[r_idx], customer, loads[r_idx], durations[r_idx])
                if pos is not None and cost < best_cost:
                    best_cost = cost
                    best_customer = customer
                    best_pos = pos
            if best_customer is not None:
                _apply_insert(r_idx, best_customer, best_pos, routes, loads, durations, instance)
                remaining.remove(best_customer)
                improved = True
        r_idx += 1
    return remaining


def _build_eip(
    routes: list[list[int]],
    pending: list[int],
    loads: list[float],
    durations: list[float],
    instance: Instance,
    criterion_fn,
) -> list[int]:
    """
    EIP — Estratégia de Inserção Paralela: melhor posição entre TODAS as rotas.
    A cada passo, insere o (cliente, rota, posição) com menor custo global.
    Quando nenhuma inserção for viável em qualquer rota, adiciona veículo extra (T2.4).
    """
    remaining = list(pending)
    while remaining:
        best_customer, best_r_idx, best_pos, best_cost = None, None, None, float("inf")
        for r_idx in range(len(routes)):
            for customer in remaining:
                pos, cost = criterion_fn(routes[r_idx], customer, loads[r_idx], durations[r_idx])
                if pos is not None and cost < best_cost:
                    best_cost = cost
                    best_customer = customer
                    best_r_idx = r_idx
                    best_pos = pos
        if best_customer is None:
            # travado: nenhuma inserção viável → abre veículo extra (T2.4)
            _add_extra_vehicle(routes, loads, durations, remaining, instance)
            continue
        _apply_insert(best_r_idx, best_customer, best_pos, routes, loads, durations, instance)
        remaining.remove(best_customer)
    return remaining


def build_solution(
    instance: Instance,
    rng: np.random.Generator,
) -> tuple[list[list[int]], list[int]]:
    """
    Algoritmo 3 completo: semeadura + inserção com critério e estratégia sorteados.

    Sorteia:
      - critério: CIMBM (com γ ∈ GAMMA_VALUES) ou CIMP
      - estratégia: EIS (sequencial) ou EIP (paralela)

    Retorna (routes, pending) onde pending é sempre [] após T2.4.
    Se routes tiver mais de K rotas não-vazias, veículos extras foram usados
    e Solution.validate() sinalizará a solução como inválida.
    """
    routes, pending = seed_routes(instance)

    nodes = instance.nodes
    t     = instance.time_matrix
    c     = instance.cost_matrix

    loads     = [float(nodes[r[0]].demanda) for r in routes]
    durations = [float(t[0][r[0]] + t[r[0]][0]) for r in routes]

    use_cimbm = bool(rng.integers(0, 2))
    use_eis   = bool(rng.integers(0, 2))
    gamma     = float(rng.choice(GAMMA_VALUES)) if use_cimbm else 0.0

    if use_cimbm:
        def pos_cost_fn(i, k, j):
            return float(c[i][k] + c[k][j] - c[i][j]) - gamma * float(c[0][k] + c[k][0])
    else:
        def pos_cost_fn(i, k, j):
            return float(c[i][k] + c[k][j] - c[i][j])

    def criterion_fn(route, customer, load_r, dur_r):
        return _best_feasible_pos(route, customer, load_r, dur_r, instance, pos_cost_fn)

    if use_eis:
        pending = _build_eis(routes, pending, loads, durations, instance, criterion_fn)
    else:
        pending = _build_eip(routes, pending, loads, durations, instance, criterion_fn)

    return routes, pending
