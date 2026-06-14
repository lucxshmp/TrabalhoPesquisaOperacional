"""
Algoritmo 3 do artigo: construção de solução inicial para o ILS-RVND.

Fase de implementação:
  T2.1 - seed_routes: semeadura por maior demanda          ← implementado
  T2.2 - best_insertion_*: critérios CIMBM e CIMP         ← a implementar
  T2.3 - build_solution: estratégias EIS e EIP            ← a implementar
  T2.4 - regra do veículo extra                           ← a implementar
"""

from __future__ import annotations
from data.structures import Instance


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
