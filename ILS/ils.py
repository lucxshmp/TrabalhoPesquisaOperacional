"""
ILS — Iterated Local Search (Algoritmo 2; Penna et al., 2013).

Junta as peças das fases anteriores:
  - construção da solução inicial (T2, `build_solution`);
  - busca local RVND (T4.1, `rvnd`);
  - perturbação sorteada (T4.2, `perturb`).

Estrutura (Algoritmo 2):

    s* = ∅
    repete MaxIter vezes:
        s  = solução inicial; s = RVND(s)
        s' = s                              # melhor do reinício corrente
        iterILS = 0
        enquanto iterILS < MaxIterILS:
            s = RVND(Perturb(s'))
            se f(s) < f(s'):  s' = s; iterILS = 0
            senão:            iterILS += 1
        se f(s') < f(s*):  s* = s'
    retorna s*

`MaxIter = 10` e `MaxIterILS = n + v` (n = nº de clientes, v = K) são os
valores do artigo. O RNG (semente via `--seed`) é propagado a construção,
RVND e perturbação, garantindo reprodutibilidade.

Registra um histórico (tempo decorrido, custo da melhor solução global) para a
curva de convergência (T5).
"""

from __future__ import annotations

import time

import numpy as np

from solution import Solution
from construction import build_solution
from rvnd import rvnd
from perturbation import perturb


def ils(
    instance,
    rng: np.random.Generator,
    maxiter: int = 10,
    maxiterils: int | None = None,
    history: list | None = None,
) -> tuple[Solution, list]:
    """
    Executa o ILS-RVND e retorna (melhor_solução, histórico).

    history: lista de tuplas (tempo_decorrido_s, custo_melhor_global), populada
             ao longo da busca para a curva de convergência.
    """
    n = len(instance.nodes) - 1
    v = instance.K
    if maxiterils is None:
        maxiterils = n + v
    if history is None:
        history = []

    t0 = time.perf_counter()
    best: Solution | None = None

    def record() -> None:
        history.append((time.perf_counter() - t0, best.cost))

    for _ in range(maxiter):
        routes, _ = build_solution(instance, rng)
        s = Solution.from_routes(instance, routes)

        # no 1º reinício, registra o custo da solução CONSTRUÍDA (antes do RVND)
        # como ponto inicial — assim a curva mostra a queda construção→RVND→ILS
        if best is None:
            best = s.copy()
            record()

        rvnd(s, rng)
        s_best = s                       # melhor solução deste reinício (s')
        if s_best.cost < best.cost:
            best = s_best.copy()
        record()

        it = 0
        while it < maxiterils:
            s = s_best.copy()
            perturb(s, rng)
            rvnd(s, rng)
            if s.cost < s_best.cost - 1e-9:
                s_best = s
                it = 0
                if s_best.cost < best.cost:
                    best = s_best.copy()
            else:
                it += 1
            record()                     # série densa do melhor-global ao longo do tempo

    return best, history
