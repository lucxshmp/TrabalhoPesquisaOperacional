"""
Algoritmo 3 do artigo: construção de solução inicial para o ILS-RVND.

Fase de implementação:
  T2.1 - seed_routes: semeadura por maior demanda          ← implementado
  T2.2 - best_insertion_*: critérios CIMBM e CIMP         ← implementado
  T2.3 - build_solution: estratégias EIS e EIP            ← a implementar
  T2.4 - regra do veículo extra                           ← a implementar
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
