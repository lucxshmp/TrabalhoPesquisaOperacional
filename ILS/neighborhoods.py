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


# ===========================================================================
# T3.2 — Vizinhanças INTER-ROTAS (6)
# ===========================================================================
#
# Diferença essencial em relação às intrarrotas: clientes migram entre DUAS
# rotas, portanto **carga e duração das duas rotas mudam** e ambas precisam
# ser reverificadas (Q e Y). Os arcos internos de um segmento que apenas muda
# de rota não alteram o custo TOTAL (cancelam-se entre as duas rotas), mas
# alteram a duração de cada rota individualmente — por isso são contabilizados
# nos deltas por rota. A matriz é simétrica, logo inverter um segmento não muda
# o custo de seus arcos internos: a orientação só afeta os arcos de fronteira.
#
# Para não abrir veículos novos durante a busca local, o destino de um Shift
# é sempre uma rota NÃO-vazia (a contagem de rotas não-vazias nunca aumenta;
# pode diminuir se a rota de origem esvaziar).


def _demands(sol: Solution):
    """Vetor de demandas indexado por nó (depósito incluso, demanda 0)."""
    return [nd.demanda for nd in sol.instance.nodes]


def _internal(seg, m):
    """Soma dos arcos internos de um segmento sobre a matriz m (custo ou tempo)."""
    s = 0.0
    for k in range(len(seg) - 1):
        s += m[seg[k]][seg[k + 1]]
    return s


# ---------------------------------------------------------------------------
# Shift(λ, 0): move um segmento de λ clientes da rota r1 para a rota r2
# ---------------------------------------------------------------------------

def _apply_best_shift(sol: Solution, lam: int) -> bool:
    c, t = sol.cost_matrix, sol.time_matrix
    Q, Y = sol.instance.Q, sol.instance.Y
    dem = _demands(sol)
    routes = sol.routes
    orientations = (False,) if lam == 1 else (False, True)  # Shift(2,0): 2 ordens do arco

    best = None  # (delta_c, r1, a, r2, m, rev, d_t_r1, d_t_r2, load_seg)
    nr = len(routes)
    for r1 in range(nr):
        route1 = routes[r1]
        if len(route1) < lam:
            continue
        full1 = [0] + route1 + [0]
        for a in range(len(route1) - lam + 1):
            prev1, first1 = full1[a], full1[a + 1]
            last1, next1 = full1[a + lam], full1[a + lam + 1]
            seg = route1[a:a + lam]
            int_c, int_t = _internal(seg, c), _internal(seg, t)
            load_seg = sum(dem[v] for v in seg)
            # remoção em r1 (sempre reduz carga e duração → r1 fica viável)
            d_c_r1 = c[prev1][next1] - c[prev1][first1] - c[last1][next1] - int_c
            d_t_r1 = t[prev1][next1] - t[prev1][first1] - t[last1][next1] - int_t

            for r2 in range(nr):
                if r2 == r1:
                    continue
                route2 = routes[r2]
                if not route2:          # não abrir veículo novo na busca local
                    continue
                if sol.loads[r2] + load_seg > Q + EPS:
                    continue
                full2 = [0] + route2 + [0]
                for m in range(len(route2) + 1):
                    i, j = full2[m], full2[m + 1]
                    for rev in orientations:
                        head, tail = (last1, first1) if rev else (first1, last1)
                        d_c_r2 = c[i][head] + c[tail][j] - c[i][j] + int_c
                        delta_c = d_c_r1 + d_c_r2
                        if delta_c < -EPS and (best is None or delta_c < best[0]):
                            d_t_r2 = t[i][head] + t[tail][j] - t[i][j] + int_t
                            if sol.durations[r2] + d_t_r2 <= Y + EPS:
                                best = (delta_c, r1, a, r2, m, rev, d_t_r1, d_t_r2, load_seg)

    if best is None:
        return False

    delta_c, r1, a, r2, m, rev, d_t_r1, d_t_r2, load_seg = best
    seg = routes[r1][a:a + lam]
    if rev:
        seg = seg[::-1]
    del routes[r1][a:a + lam]
    routes[r2][m:m] = seg
    sol.cost += delta_c
    sol.durations[r1] += d_t_r1
    sol.durations[r2] += d_t_r2
    sol.loads[r1] -= load_seg
    sol.loads[r2] += load_seg
    return True


def shift10(sol: Solution) -> bool:
    """Shift(1,0): move 1 cliente de uma rota para outra (melhor posição)."""
    return _apply_best_shift(sol, 1)


def shift20(sol: Solution) -> bool:
    """Shift(2,0): move 2 clientes consecutivos, testando o arco nas 2 ordens."""
    return _apply_best_shift(sol, 2)


# ---------------------------------------------------------------------------
# Swap(λ1, λ2): troca um segmento de λ1 clientes de r1 por λ2 clientes de r2
# ---------------------------------------------------------------------------

def _apply_best_swap(sol: Solution, lam1: int, lam2: int) -> bool:
    c, t = sol.cost_matrix, sol.time_matrix
    Q, Y = sol.instance.Q, sol.instance.Y
    dem = _demands(sol)
    routes = sol.routes
    or1 = (False,) if lam1 == 1 else (False, True)
    or2 = (False,) if lam2 == 1 else (False, True)

    best = None  # (delta_c, r1, a, r2, b, rev1, rev2, d_t_r1, d_t_r2, load1seg, load2seg)
    nr = len(routes)
    for r1 in range(nr):
        route1 = routes[r1]
        if len(route1) < lam1:
            continue
        full1 = [0] + route1 + [0]
        for a in range(len(route1) - lam1 + 1):
            prev1, first1 = full1[a], full1[a + 1]
            last1, next1 = full1[a + lam1], full1[a + lam1 + 1]
            seg1 = route1[a:a + lam1]
            int_c1, int_t1 = _internal(seg1, c), _internal(seg1, t)
            load1seg = sum(dem[v] for v in seg1)

            for r2 in range(nr):
                if r2 == r1:
                    continue
                route2 = routes[r2]
                if len(route2) < lam2:
                    continue
                full2 = [0] + route2 + [0]
                for b in range(len(route2) - lam2 + 1):
                    prev2, first2 = full2[b], full2[b + 1]
                    last2, next2 = full2[b + lam2], full2[b + lam2 + 1]
                    seg2 = route2[b:b + lam2]
                    int_c2, int_t2 = _internal(seg2, c), _internal(seg2, t)
                    load2seg = sum(dem[v] for v in seg2)
                    # capacidade das duas rotas após a troca
                    if sol.loads[r1] - load1seg + load2seg > Q + EPS:
                        continue
                    if sol.loads[r2] - load2seg + load1seg > Q + EPS:
                        continue
                    for rev1 in or1:                       # orientação de seg1 (vai p/ r2)
                        h1, t1 = (last1, first1) if rev1 else (first1, last1)
                        for rev2 in or2:                   # orientação de seg2 (vai p/ r1)
                            h2, t2 = (last2, first2) if rev2 else (first2, last2)
                            d_c_r1 = ((c[prev1][h2] + int_c2 + c[t2][next1])
                                      - (c[prev1][first1] + int_c1 + c[last1][next1]))
                            d_c_r2 = ((c[prev2][h1] + int_c1 + c[t1][next2])
                                      - (c[prev2][first2] + int_c2 + c[last2][next2]))
                            delta_c = d_c_r1 + d_c_r2
                            if delta_c < -EPS and (best is None or delta_c < best[0]):
                                d_t_r1 = ((t[prev1][h2] + int_t2 + t[t2][next1])
                                          - (t[prev1][first1] + int_t1 + t[last1][next1]))
                                d_t_r2 = ((t[prev2][h1] + int_t1 + t[t1][next2])
                                          - (t[prev2][first2] + int_t2 + t[last2][next2]))
                                if (sol.durations[r1] + d_t_r1 <= Y + EPS
                                        and sol.durations[r2] + d_t_r2 <= Y + EPS):
                                    best = (delta_c, r1, a, r2, b, rev1, rev2,
                                            d_t_r1, d_t_r2, load1seg, load2seg)

    if best is None:
        return False

    delta_c, r1, a, r2, b, rev1, rev2, d_t_r1, d_t_r2, load1seg, load2seg = best
    route1, route2 = routes[r1], routes[r2]
    seg1 = route1[a:a + lam1]
    seg2 = route2[b:b + lam2]
    s1 = seg1[::-1] if rev1 else seg1
    s2 = seg2[::-1] if rev2 else seg2
    routes[r1] = route1[:a] + s2 + route1[a + lam1:]
    routes[r2] = route2[:b] + s1 + route2[b + lam2:]
    sol.cost += delta_c
    sol.durations[r1] += d_t_r1
    sol.durations[r2] += d_t_r2
    sol.loads[r1] += load2seg - load1seg
    sol.loads[r2] += load1seg - load2seg
    return True


def swap11(sol: Solution) -> bool:
    """Swap(1,1): troca 1 cliente de r1 por 1 cliente de r2."""
    return _apply_best_swap(sol, 1, 1)


def swap21(sol: Solution) -> bool:
    """Swap(2,1): troca 2 clientes de r1 por 1 de r2 (seg de 2 nas 2 ordens)."""
    return _apply_best_swap(sol, 2, 1)


def swap22(sol: Solution) -> bool:
    """Swap(2,2): troca 2 clientes de r1 por 2 de r2 (4 combinações de arcos)."""
    return _apply_best_swap(sol, 2, 2)


# ---------------------------------------------------------------------------
# Cross: troca os sufixos (terminando no depósito) de duas rotas
# ---------------------------------------------------------------------------

def _cross_arrays(route, t, dem):
    """
    Pré-computa, para uma rota, arrays de duração/carga de prefixos e sufixos,
    de modo que cada candidato Cross seja avaliado em O(1).

      pt[k] = duração do caminho aberto depósito → route[0] → … → route[k-1]
      st[k] = duração do caminho aberto route[k] → … → route[-1] → depósito
      pl[k] = soma das demandas de route[:k]
      sl[k] = soma das demandas de route[k:]
    (pt[0]=st[n]=pl[0]=sl[n]=0)
    """
    n = len(route)
    full = [0] + route + [0]
    pt = [0.0] * (n + 1)
    pl = [0.0] * (n + 1)
    for k in range(1, n + 1):
        pt[k] = pt[k - 1] + t[full[k - 1]][full[k]]
        pl[k] = pl[k - 1] + dem[route[k - 1]]
    st = [0.0] * (n + 1)
    sl = [0.0] * (n + 1)
    for k in range(n - 1, -1, -1):
        st[k] = st[k + 1] + t[full[k + 1]][full[k + 2]]
        sl[k] = sl[k + 1] + dem[route[k]]
    return full, pt, pl, st, sl


def cross(sol: Solution) -> bool:
    """
    Cross: corta r1 após a posição cut1 e r2 após cut2, e troca os sufixos.

        r1 = head1 + tail1     r2 = head2 + tail2
        →   head1 + tail2          head2 + tail1

    Só os 2 arcos de emenda mudam, logo o delta de custo é O(1):
        Δ = c[h1last][t2first] + c[h2last][t1first]
          − c[h1last][t1first] − c[h2last][t2first]
    Cargas/durações das novas rotas vêm dos arrays de prefixo/sufixo (O(1)).
    """
    c, t = sol.cost_matrix, sol.time_matrix
    Q, Y = sol.instance.Q, sol.instance.Y
    dem = _demands(sol)
    routes = sol.routes
    nr = len(routes)

    info = [_cross_arrays(routes[r], t, dem) for r in range(nr)]

    best = None  # (delta_c, r1, cut1, r2, cut2, nd1, nd2, nl1, nl2)
    for r1 in range(nr):
        full1, pt1, pl1, st1, sl1 = info[r1]
        len1 = len(routes[r1])
        for r2 in range(r1 + 1, nr):
            full2, pt2, pl2, st2, sl2 = info[r2]
            len2 = len(routes[r2])
            for cut1 in range(len1 + 1):
                a1last, t1first = full1[cut1], full1[cut1 + 1]
                for cut2 in range(len2 + 1):
                    a2last, t2first = full2[cut2], full2[cut2 + 1]
                    delta_c = (c[a1last][t2first] + c[a2last][t1first]
                               - c[a1last][t1first] - c[a2last][t2first])
                    if delta_c < -EPS and (best is None or delta_c < best[0]):
                        nl1 = pl1[cut1] + sl2[cut2]
                        nl2 = pl2[cut2] + sl1[cut1]
                        if nl1 <= Q + EPS and nl2 <= Q + EPS:
                            nd1 = pt1[cut1] + t[a1last][t2first] + st2[cut2]
                            nd2 = pt2[cut2] + t[a2last][t1first] + st1[cut1]
                            if nd1 <= Y + EPS and nd2 <= Y + EPS:
                                best = (delta_c, r1, cut1, r2, cut2, nd1, nd2, nl1, nl2)

    if best is None:
        return False

    delta_c, r1, cut1, r2, cut2, nd1, nd2, nl1, nl2 = best
    route1, route2 = routes[r1], routes[r2]
    head1, tail1 = route1[:cut1], route1[cut1:]
    head2, tail2 = route2[:cut2], route2[cut2:]
    routes[r1] = head1 + tail2
    routes[r2] = head2 + tail1
    sol.cost += delta_c
    sol.durations[r1], sol.durations[r2] = nd1, nd2
    sol.loads[r1], sol.loads[r2] = nl1, nl2
    return True


# Registro das vizinhanças inter-rotas — usado pela lista LV do RVND (T4).
INTER_NEIGHBORHOODS = [shift10, shift20, swap11, swap21, swap22, cross]
