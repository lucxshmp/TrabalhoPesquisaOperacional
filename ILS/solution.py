import copy
import numpy as np


class Solution:
    """
    Representa uma solução do VRP.

    routes: lista de rotas, cada rota é uma lista de IDs de nós clientes
            (depósito 0 é implícito no início e no fim de cada rota).
    cost:   custo total (soma das distâncias de todas as rotas).
    loads:  carga acumulada por rota (sum das demandas).
    durations: duração total por rota (tempo de viagem arco a arco).
    """

    def __init__(self, instance):
        self.instance = instance
        self.cost_matrix = np.array(instance.cost_matrix)
        self.time_matrix = np.array(instance.time_matrix)
        self.routes: list[list[int]] = []
        self.cost: float = 0.0
        self.loads: list[float] = []
        self.durations: list[float] = []

    # ------------------------------------------------------------------
    # Construção a partir de uma lista de rotas já definida
    # ------------------------------------------------------------------

    @classmethod
    def from_routes(cls, instance, routes: list[list[int]]) -> "Solution":
        sol = cls(instance)
        sol.routes = [list(r) for r in routes]
        sol.recompute()
        return sol

    # ------------------------------------------------------------------
    # Cópia profunda (usada no ILS para preservar melhor solução)
    # ------------------------------------------------------------------

    def copy(self) -> "Solution":
        sol = Solution(self.instance)
        sol.cost_matrix = self.cost_matrix
        sol.time_matrix = self.time_matrix
        sol.routes = [list(r) for r in self.routes]
        sol.cost = self.cost
        sol.loads = list(self.loads)
        sol.durations = list(self.durations)
        return sol

    # ------------------------------------------------------------------
    # Recomputação completa (O(n) — usada após construção e em debug)
    # ------------------------------------------------------------------

    def recompute(self) -> None:
        c = self.cost_matrix
        t = self.time_matrix
        nodes = self.instance.nodes

        total_cost = 0.0
        loads = []
        durations = []

        for route in self.routes:
            if not route:
                loads.append(0.0)
                durations.append(0.0)
                continue

            load = sum(nodes[v].demanda for v in route)
            dur = t[0][route[0]]
            cost_r = c[0][route[0]]
            for k in range(len(route) - 1):
                cost_r += c[route[k]][route[k + 1]]
                dur += t[route[k]][route[k + 1]]
            cost_r += c[route[-1]][0]
            dur += t[route[-1]][0]

            total_cost += cost_r
            loads.append(load)
            durations.append(dur)

        self.cost = total_cost
        self.loads = loads
        self.durations = durations

    # ------------------------------------------------------------------
    # Utilitários de custo por rota (usados em vizinhanças)
    # ------------------------------------------------------------------

    def route_cost(self, route_idx: int) -> float:
        route = self.routes[route_idx]
        if not route:
            return 0.0
        c = self.cost_matrix
        total = c[0][route[0]]
        for k in range(len(route) - 1):
            total += c[route[k]][route[k + 1]]
        total += c[route[-1]][0]
        return float(total)

    def route_duration(self, route_idx: int) -> float:
        route = self.routes[route_idx]
        if not route:
            return 0.0
        t = self.time_matrix
        total = t[0][route[0]]
        for k in range(len(route) - 1):
            total += t[route[k]][route[k + 1]]
        total += t[route[-1]][0]
        return float(total)

    def route_load(self, route_idx: int) -> float:
        nodes = self.instance.nodes
        return float(sum(nodes[v].demanda for v in self.routes[route_idx]))

    # ------------------------------------------------------------------
    # Modo debug (T3.3): consistência dos deltas
    # ------------------------------------------------------------------

    def assert_consistent(self, tol: float = 1e-6, context: str = "") -> None:
        """
        Recomputa custo, cargas e durações do zero e compara com os
        acumuladores incrementais (cost/loads/durations), disparando
        AssertionError no primeiro desvio.

        Usado após cada movimento das vizinhanças quando `--debug` está ativo,
        para pegar erros de delta cedo (diferente de validate(), aqui só se
        verifica a CONSISTÊNCIA dos incrementais, não a viabilidade).
        """
        c = self.cost_matrix
        t = self.time_matrix
        nodes = self.instance.nodes
        where = f" [{context}]" if context else ""

        assert len(self.loads) == len(self.routes), (
            f"loads tem {len(self.loads)} entradas para "
            f"{len(self.routes)} rotas{where}"
        )
        assert len(self.durations) == len(self.routes), (
            f"durations tem {len(self.durations)} entradas para "
            f"{len(self.routes)} rotas{where}"
        )

        total_cost = 0.0
        for r_idx, route in enumerate(self.routes):
            if not route:
                assert abs(self.loads[r_idx]) <= tol, (
                    f"rota {r_idx} vazia mas carga={self.loads[r_idx]:.6f}{where}"
                )
                assert abs(self.durations[r_idx]) <= tol, (
                    f"rota {r_idx} vazia mas duração={self.durations[r_idx]:.6f}{where}"
                )
                continue

            load = sum(nodes[v].demanda for v in route)
            cost_r = c[0][route[0]]
            dur = t[0][route[0]]
            for k in range(len(route) - 1):
                cost_r += c[route[k]][route[k + 1]]
                dur += t[route[k]][route[k + 1]]
            cost_r += c[route[-1]][0]
            dur += t[route[-1]][0]
            total_cost += cost_r

            assert abs(self.loads[r_idx] - load) <= tol, (
                f"carga incremental da rota {r_idx} ({self.loads[r_idx]:.6f}) "
                f"≠ recomputada ({load:.6f}){where}"
            )
            assert abs(self.durations[r_idx] - dur) <= tol, (
                f"duração incremental da rota {r_idx} ({self.durations[r_idx]:.6f}) "
                f"≠ recomputada ({dur:.6f}){where}"
            )

        assert abs(self.cost - total_cost) <= tol, (
            f"custo incremental ({self.cost:.6f}) "
            f"≠ recomputado ({total_cost:.6f}){where}"
        )

    # ------------------------------------------------------------------
    # Validação de corretude
    # ------------------------------------------------------------------

    def validate(self) -> tuple[bool, list[str]]:
        """
        Verifica:
          1. Cada cliente é visitado exatamente uma vez.
          2. Carga de cada rota ≤ Q.
          3. Duração de cada rota ≤ Y.
          4. Número de rotas ≤ K.
          5. Valores incrementais (cost/loads/durations) batem com recomputação.

        Retorna (ok: bool, erros: list[str]).
        """
        inst = self.instance
        n_customers = len(inst.nodes) - 1  # exclui depósito (nó 0)
        errors: list[str] = []

        # --- cobertura ---
        visited: dict[int, int] = {}
        for r_idx, route in enumerate(self.routes):
            for v in route:
                if v == 0:
                    errors.append(f"Rota {r_idx}: depósito (nó 0) aparece dentro da rota.")
                visited[v] = visited.get(v, 0) + 1

        for v in range(1, len(inst.nodes)):
            cnt = visited.get(v, 0)
            if cnt == 0:
                errors.append(f"Nó {v} não foi visitado.")
            elif cnt > 1:
                errors.append(f"Nó {v} foi visitado {cnt} vezes.")

        if len(visited) > n_customers:
            extras = [v for v in visited if v not in range(1, len(inst.nodes))]
            errors.append(f"Nós inválidos nas rotas: {extras}")

        # --- capacidade e duração ---
        for r_idx, route in enumerate(self.routes):
            if not route:
                continue
            load = self.route_load(r_idx)
            dur = self.route_duration(r_idx)
            if load > inst.Q + 1e-9:
                errors.append(
                    f"Rota {r_idx}: carga {load:.2f} excede Q={inst.Q}."
                )
            if dur > inst.Y + 1e-9:
                errors.append(
                    f"Rota {r_idx}: duração {dur:.4f}h excede Y={inst.Y}h."
                )

        # --- frota ---
        non_empty = sum(1 for r in self.routes if r)
        if non_empty > inst.K:
            errors.append(
                f"Número de rotas não-vazias ({non_empty}) excede K={inst.K}."
            )

        # --- consistência dos valores incrementais ---
        tol = 1e-6
        for r_idx in range(len(self.routes)):
            expected_load = self.route_load(r_idx)
            if abs(self.loads[r_idx] - expected_load) > tol:
                errors.append(
                    f"Rota {r_idx}: loads[{r_idx}]={self.loads[r_idx]:.6f} "
                    f"difere do valor recomputado {expected_load:.6f}."
                )
            expected_dur = self.route_duration(r_idx)
            if abs(self.durations[r_idx] - expected_dur) > tol:
                errors.append(
                    f"Rota {r_idx}: durations[{r_idx}]={self.durations[r_idx]:.6f} "
                    f"difere do valor recomputado {expected_dur:.6f}."
                )

        expected_cost = sum(self.route_cost(i) for i in range(len(self.routes)))
        if abs(self.cost - expected_cost) > tol:
            errors.append(
                f"Custo total {self.cost:.6f} difere do valor recomputado {expected_cost:.6f}."
            )

        return len(errors) == 0, errors

    # ------------------------------------------------------------------
    # Representação
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n_routes = sum(1 for r in self.routes if r)
        return f"Solution(cost={self.cost:.4f}, routes={n_routes}/{self.instance.K})"
