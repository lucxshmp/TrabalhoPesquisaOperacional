"""
Perturbações do ILS (Algoritmo 2; Penna et al., 2013).

Duas perturbações, sorteadas a cada chamada:

  - Multiple-Shift(1,1): vários Shift(1,0) aleatórios — move 1 cliente de uma
    rota para outra, ambos sorteados;
  - Multiple-Swap(1,1): vários Swap(1,1) aleatórios — troca 1 cliente de uma
    rota por 1 cliente de outra, tudo sorteado.

Diferença em relação às vizinhanças (neighborhoods.py): aqui os movimentos são
ALEATÓRIOS (diversificação), não best-improvement. Cada movimento é aplicado
apenas se mantiver a viabilidade (Q e Y) — a solução resultante continua viável
e com cost/loads/durations consistentes (validate/assert_consistent passam).
A intensidade (nº de movimentos) é sorteada quando não informada.
"""

from __future__ import annotations

import numpy as np

from solution import Solution

EPS = 1e-9
MAX_TRIES = 20  # tentativas p/ achar um movimento aleatório viável por passo


def _demands(sol: Solution):
    return [nd.demanda for nd in sol.instance.nodes]


def _nonempty(routes):
    return [i for i, r in enumerate(routes) if r]


def _random_shift(sol: Solution, rng, dem, Q, Y) -> bool:
    """Tenta UM Shift(1,0) aleatório viável (cliente de r1 → r2). True se aplicou."""
    routes = sol.routes
    ne = _nonempty(routes)
    if len(ne) < 2:
        return False
    r1 = int(rng.choice(ne))
    r2 = int(rng.choice([i for i in ne if i != r1]))   # destino não-vazio
    route1, route2 = routes[r1], routes[r2]
    a = int(rng.integers(0, len(route1)))
    k = route1[a]
    if sol.loads[r2] + dem[k] > Q + EPS:
        return False

    c, t = sol.cost_matrix, sol.time_matrix
    full1 = [0] + route1 + [0]
    prev1, nxt1 = full1[a], full1[a + 2]
    d_c_r1 = c[prev1][nxt1] - c[prev1][k] - c[k][nxt1]
    d_t_r1 = t[prev1][nxt1] - t[prev1][k] - t[k][nxt1]

    m = int(rng.integers(0, len(route2) + 1))
    full2 = [0] + route2 + [0]
    i, j = full2[m], full2[m + 1]
    d_t_r2 = t[i][k] + t[k][j] - t[i][j]
    if sol.durations[r2] + d_t_r2 > Y + EPS:
        return False
    d_c_r2 = c[i][k] + c[k][j] - c[i][j]

    del route1[a]
    route2.insert(m, k)
    sol.cost += d_c_r1 + d_c_r2
    sol.durations[r1] += d_t_r1
    sol.durations[r2] += d_t_r2
    sol.loads[r1] -= dem[k]
    sol.loads[r2] += dem[k]
    return True


def _random_swap(sol: Solution, rng, dem, Q, Y) -> bool:
    """Tenta UM Swap(1,1) aleatório viável (cliente de r1 ↔ cliente de r2)."""
    routes = sol.routes
    ne = _nonempty(routes)
    if len(ne) < 2:
        return False
    r1 = int(rng.choice(ne))
    r2 = int(rng.choice([i for i in ne if i != r1]))
    route1, route2 = routes[r1], routes[r2]
    a = int(rng.integers(0, len(route1)))
    b = int(rng.integers(0, len(route2)))
    ka, kb = route1[a], route2[b]

    if sol.loads[r1] - dem[ka] + dem[kb] > Q + EPS:
        return False
    if sol.loads[r2] - dem[kb] + dem[ka] > Q + EPS:
        return False

    c, t = sol.cost_matrix, sol.time_matrix
    full1 = [0] + route1 + [0]
    full2 = [0] + route2 + [0]
    prev1, nxt1 = full1[a], full1[a + 2]
    prev2, nxt2 = full2[b], full2[b + 2]
    d_t_r1 = (t[prev1][kb] + t[kb][nxt1]) - (t[prev1][ka] + t[ka][nxt1])
    d_t_r2 = (t[prev2][ka] + t[ka][nxt2]) - (t[prev2][kb] + t[kb][nxt2])
    if sol.durations[r1] + d_t_r1 > Y + EPS:
        return False
    if sol.durations[r2] + d_t_r2 > Y + EPS:
        return False
    d_c_r1 = (c[prev1][kb] + c[kb][nxt1]) - (c[prev1][ka] + c[ka][nxt1])
    d_c_r2 = (c[prev2][ka] + c[ka][nxt2]) - (c[prev2][kb] + c[kb][nxt2])

    route1[a], route2[b] = kb, ka
    sol.cost += d_c_r1 + d_c_r2
    sol.durations[r1] += d_t_r1
    sol.durations[r2] += d_t_r2
    sol.loads[r1] += dem[kb] - dem[ka]
    sol.loads[r2] += dem[ka] - dem[kb]
    return True


def _apply_multiple(sol, rng, n_moves, move_fn) -> int:
    """Aplica n_moves movimentos aleatórios viáveis de move_fn. Retorna o nº aplicado."""
    dem = _demands(sol)
    Q, Y = sol.instance.Q, sol.instance.Y
    applied = 0
    for _ in range(n_moves):
        for _try in range(MAX_TRIES):
            if move_fn(sol, rng, dem, Q, Y):
                applied += 1
                break
    return applied


def multiple_shift(sol: Solution, rng, n_moves: int) -> int:
    """Multiple-Shift(1,1): n_moves relocações aleatórias viáveis entre rotas."""
    return _apply_multiple(sol, rng, n_moves, _random_shift)


def multiple_swap(sol: Solution, rng, n_moves: int) -> int:
    """Multiple-Swap(1,1): n_moves trocas aleatórias viáveis entre rotas."""
    return _apply_multiple(sol, rng, n_moves, _random_swap)


def perturb(sol: Solution, rng, n_moves: int | None = None) -> int:
    """
    Sorteia e aplica uma das duas perturbações (Multiple-Swap ou Multiple-Shift).

    Se n_moves não for informado, a intensidade é sorteada em [1, nº de rotas
    não-vazias). Retorna quantos movimentos foram efetivamente aplicados.
    """
    n_ne = sum(1 for r in sol.routes if r)
    if n_moves is None:
        n_moves = int(rng.integers(1, max(2, n_ne)))
    if bool(rng.integers(0, 2)):
        return multiple_swap(sol, rng, n_moves)
    return multiple_shift(sol, rng, n_moves)
