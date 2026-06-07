from master.columns import Column

def pricing(instance, duals):

    MAX_CLIENTES = 5
    MAX_EXPANSIONS = 30000
    MAX_NEW_COLUMNS = 40
    K_NEIGHBORS = 5

    new_columns = []
    depot = 0

    for start in instance.nodes:
        if start.id == depot:
            continue

        stack = [(
            [depot, start.id],
            start.demanda,
            instance.time_matrix[depot][start.id],
            0  # custo parcial
        )]

        expansions = 0

        while stack:

            route, demand, time, partial_cost = stack.pop()

            expansions += 1
            if expansions > MAX_EXPANSIONS:
                print("Limite de expansão atingido")
                break

            # -------------------
            # limite de clientes
            # -------------------
            if len(route) - 1 > MAX_CLIENTES:
                continue

            last = route[-1]

            # -------------------
            # fechar rota
            # -------------------
            cost = partial_cost + instance.cost_matrix[last][depot]
            total_time = time + instance.time_matrix[last][depot]

            if total_time <= instance.Y:

                dual_sum = sum(duals.get(n, 0) for n in route if n != 0)
                reduced_cost = cost - dual_sum

                if reduced_cost < -1e-6:
                    full_route = route + [depot]

                    col = Column(full_route, cost, total_time)
                    new_columns.append(col)

                    print(f"Nova rota: {full_route} | rc={reduced_cost:.2f}")

                    # limite de novas colunas
                    if len(new_columns) >= MAX_NEW_COLUMNS:
                        return new_columns

            # -------------------
            # EXPANSÃO INTELIGENTE (KNN)
            # -------------------
            if len(route) - 1 < MAX_CLIENTES:

                neighbors = sorted(
                    instance.nodes,
                    key=lambda n: instance.cost_matrix[last][n.id]
                )[:K_NEIGHBORS]

                for node in neighbors:

                    if node.id in route or node.id == depot:
                        continue

                    new_demand = demand + node.demanda
                    if new_demand > instance.Q:
                        continue

                    new_time = time + instance.time_matrix[last][node.id]
                    if new_time > instance.Y:
                        continue

                    arc_cost = instance.cost_matrix[last][node.id]
                    new_partial_cost = partial_cost + arc_cost

                    # poda simples por custo
                    if new_partial_cost > 200:
                        continue

                    new_route = route + [node.id]

                    stack.append((
                        new_route,
                        new_demand,
                        new_time,
                        new_partial_cost
                    ))

    return new_columns