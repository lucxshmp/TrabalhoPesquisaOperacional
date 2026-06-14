"""
Vizinhanças do ILS-RVND (Penna et al., 2013; Kramer, Subramanian & Penna, 2016).

T3.1 — Vizinhanças INTRARROTAS (4): Reinserção, Or-opt2, Or-opt3, Exchange.

Convenções (iguais ao resto do pacote):
  - depósito = nó 0, implícito no início e no fim de cada rota;
  - custo = `cost_matrix`; duração = `time_matrix` (distância / 40);
  - uma rota é `solution.routes[r]` = lista de clientes (sem o depósito).

Cada vizinhança:
  - avalia cada movimento candidato por **delta O(1)** (aritmética sobre os
    arcos afetados, sem recomputar a rota inteira);
  - usa **best-improvement**: aplica o melhor movimento de melhora (delta de
    custo mais negativo) encontrado em toda a solução;
  - testa a **viabilidade** (Q e Y) apenas sobre candidatos de melhora — como o
    movimento é intrarrota, a carga não muda; só a duração é reverificada;
  - atualiza `cost` e `durations[r]` de forma incremental.

Cada função retorna `True` se aplicou um movimento (a solução melhorou), ou
`False` caso nenhum movimento de melhora viável exista naquela vizinhança.
"""

from __future__ import annotations

from solution import Solution

EPS = 1e-9


# ---------------------------------------------------------------------------
# Or-opt(L): relocação de um segmento de L clientes consecutivos dentro da rota
# ---------------------------------------------------------------------------

def _best_or_opt_in_route(route, c, t, L, dur_r, Y):
    """
    Melhor movimento Or-opt(L) DENTRO de uma única rota.

    Remove o segmento de L clientes consecutivos `route[a : a+L]` (mantendo a
    orientação) e o reinsere em outra posição. Avaliação O(1) por candidato:

        remove_gain = c[prev][first] + c[last][next] − c[prev][next]
        insert_cost = c[i][first]    + c[last][j]    − c[i][j]
        delta       = insert_cost − remove_gain

    Retorna (delta_c, delta_d, a, ins) do melhor movimento viável de melhora,
    ou None. `ins` é a posição de inserção na rota já sem o segmento.
    """
    n = len(route)
    if n <= L:
        return None

    full = [0] + route + [0]  # depósito implícito nas pontas
    best = None  # (delta_c, delta_d, a, ins)

    for a in range(n - L + 1):
        # segmento ocupa full[a+1 .. a+L]; vizinhos no full: prev=full[a], next=full[a+L+1]
        prev, first = full[a], full[a + 1]
        last, nxt = full[a + L], full[a + L + 1]

        rem_c = c[prev][first] + c[last][nxt] - c[prev][nxt]
        rem_d = t[prev][first] + t[last][nxt] - t[prev][nxt]

        # rota (em coords full) já sem o segmento removido
        rest_full = full[: a + 1] + full[a + L + 1:]

        for ins in range(len(rest_full) - 1):
            if ins == a:
                continue  # reinsere exatamente onde estava (movimento nulo)
            i, j = rest_full[ins], rest_full[ins + 1]
            add_c = c[i][first] + c[last][j] - c[i][j]
            delta_c = add_c - rem_c
            if delta_c < -EPS and (best is None or delta_c < best[0]):
                add_d = t[i][first] + t[last][j] - t[i][j]
                delta_d = add_d - rem_d
                if dur_r + delta_d <= Y + EPS:  # viabilidade de duração
                    best = (delta_c, delta_d, a, ins)

    return best


def _apply_best_or_opt(sol: Solution, L: int) -> bool:
    """Aplica o melhor Or-opt(L) de melhora em toda a solução (best-improvement)."""
    c, t = sol.cost_matrix, sol.time_matrix
    Y = sol.instance.Y

    best = None  # (delta_c, delta_d, r_idx, a, ins)
    for r_idx, route in enumerate(sol.routes):
        m = _best_or_opt_in_route(route, c, t, L, sol.durations[r_idx], Y)
        if m is not None and (best is None or m[0] < best[0]):
            best = (m[0], m[1], r_idx, m[2], m[3])

    if best is None:
        return False

    delta_c, delta_d, r_idx, a, ins = best
    route = sol.routes[r_idx]
    seg = route[a:a + L]
    rest = route[:a] + route[a + L:]
    sol.routes[r_idx] = rest[:ins] + seg + rest[ins:]
    sol.cost += delta_c
    sol.durations[r_idx] += delta_d
    return True


def reinsertion(sol: Solution) -> bool:
    """Reinserção (Or-opt de 1 cliente): move 1 cliente para a melhor posição."""
    return _apply_best_or_opt(sol, 1)


def or_opt2(sol: Solution) -> bool:
    """Or-opt2: move um segmento de 2 clientes consecutivos."""
    return _apply_best_or_opt(sol, 2)


def or_opt3(sol: Solution) -> bool:
    """Or-opt3: move um segmento de 3 clientes consecutivos."""
    return _apply_best_or_opt(sol, 3)


# ---------------------------------------------------------------------------
# Exchange: troca de dois clientes dentro da mesma rota
# ---------------------------------------------------------------------------

def _best_exchange_in_route(route, c, t, dur_r, Y):
    """
    Melhor troca (swap) de dois clientes DENTRO de uma única rota.

    Avaliação O(1) por candidato sobre os arcos afetados. Há dois casos:
      - adjacentes (y == x+1): os clientes compartilham um arco interno;
      - não-adjacentes: dois pares de arcos independentes.

    Retorna (delta_c, delta_d, x, y) do melhor swap viável de melhora, ou None.
    """
    n = len(route)
    if n < 2:
        return None

    full = [0] + route + [0]
    best = None  # (delta_c, delta_d, x, y)

    for x in range(n):
        na = full[x + 1]
        pa = full[x]          # antecessor de na
        for y in range(x + 1, n):
            nb = full[y + 1]
            sb = full[y + 2]  # sucessor de nb

            if y == x + 1:
                # adjacentes: arco na→nb é interno (não duplicar)
                rem_c = c[pa][na] + c[na][nb] + c[nb][sb]
                add_c = c[pa][nb] + c[nb][na] + c[na][sb]
                rem_d = t[pa][na] + t[na][nb] + t[nb][sb]
                add_d = t[pa][nb] + t[nb][na] + t[na][sb]
            else:
                sa = full[x + 2]  # sucessor de na
                pb = full[y]      # antecessor de nb
                rem_c = c[pa][na] + c[na][sa] + c[pb][nb] + c[nb][sb]
                add_c = c[pa][nb] + c[nb][sa] + c[pb][na] + c[na][sb]
                rem_d = t[pa][na] + t[na][sa] + t[pb][nb] + t[nb][sb]
                add_d = t[pa][nb] + t[nb][sa] + t[pb][na] + t[na][sb]

            delta_c = add_c - rem_c
            if delta_c < -EPS and (best is None or delta_c < best[0]):
                delta_d = add_d - rem_d
                if dur_r + delta_d <= Y + EPS:
                    best = (delta_c, delta_d, x, y)

    return best


def exchange(sol: Solution) -> bool:
    """Exchange intrarrota: troca dois clientes (best-improvement em toda a solução)."""
    c, t = sol.cost_matrix, sol.time_matrix
    Y = sol.instance.Y

    best = None  # (delta_c, delta_d, r_idx, x, y)
    for r_idx, route in enumerate(sol.routes):
        m = _best_exchange_in_route(route, c, t, sol.durations[r_idx], Y)
        if m is not None and (best is None or m[0] < best[0]):
            best = (m[0], m[1], r_idx, m[2], m[3])

    if best is None:
        return False

    delta_c, delta_d, r_idx, x, y = best
    route = sol.routes[r_idx]
    route[x], route[y] = route[y], route[x]
    sol.cost += delta_c
    sol.durations[r_idx] += delta_d
    return True


# Registro das vizinhanças intrarrotas — usado pela lista LV2 do RVND (T4).
INTRA_NEIGHBORHOODS = [reinsertion, or_opt2, or_opt3, exchange]
