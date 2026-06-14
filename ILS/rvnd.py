"""
RVND — Random Variable Neighborhood Descent (Algoritmo 4 do artigo).

Estrutura (Penna et al., 2013; Kramer, Subramanian & Penna, 2016):

  - LV  = lista de vizinhanças INTER-ROTAS (T3.2);
  - LV2 = lista de vizinhanças INTRARROTAS (T3.1).

Laço externo (sobre LV):
  1. sorteia uma vizinhança de LV e faz best-improvement nela;
  2. se melhorou → roda a busca intrarrotas (RVND sobre LV2 até o ótimo local
     intra) e **reinicializa LV** (todas as vizinhanças voltam a ser candidatas);
  3. se não melhorou → **remove** essa vizinhança de LV.
  Termina quando LV fica vazia: nenhuma inter melhora e a solução está em ótimo
  local intra → ótimo local em relação às 10 vizinhanças.

Cada vizinhança já aplica apenas movimentos viáveis (Q e Y) e de melhora
(custo nunca piora), com avaliação por delta O(1) e atualização incremental.
"""

from __future__ import annotations

import numpy as np

from solution import Solution
from neighborhoods import INTER_NEIGHBORHOODS, INTRA_NEIGHBORHOODS


def _rvnd_over(sol: Solution, neighborhoods, rng: np.random.Generator) -> bool:
    """
    RVND sobre uma única lista de vizinhanças (usado para LV2, intrarrotas).

    Sorteia uma vizinhança da lista corrente; se melhora, reinicializa a lista;
    senão remove-a. Retorna True se aplicou ao menos um movimento.
    """
    lv = list(neighborhoods)
    improved_any = False
    while lv:
        idx = int(rng.integers(0, len(lv)))
        if lv[idx](sol):
            improved_any = True
            lv = list(neighborhoods)   # reinicializa a lista
        else:
            lv.pop(idx)                # remove a vizinhança esgotada
    return improved_any


def local_search_intra(sol: Solution, rng: np.random.Generator) -> bool:
    """Busca local intrarrotas (LV2) até o ótimo local intra."""
    return _rvnd_over(sol, INTRA_NEIGHBORHOODS, rng)


def rvnd(sol: Solution, rng: np.random.Generator) -> Solution:
    """
    Algoritmo 4: desce até um ótimo local sobre as 10 vizinhanças, modificando
    `sol` no lugar. Retorna a própria `sol` (já em ótimo local).
    """
    lv = list(INTER_NEIGHBORHOODS)
    while lv:
        idx = int(rng.integers(0, len(lv)))
        if lv[idx](sol):
            local_search_intra(sol, rng)      # LV2 + reinicializa LV
            lv = list(INTER_NEIGHBORHOODS)
        else:
            lv.pop(idx)
    return sol
