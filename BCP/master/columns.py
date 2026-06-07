class Column:
    def __init__(self, route, cost, time):
        self.route = route
        self.cost = cost
        self.coverage = set(route[1:-1])
        self.time = time

    @classmethod
    def create_initial_routes(cls, instance):
        columns = []
        depot = 0

        for node in instance.nodes:

            if node.id == depot:
                continue
            
            if node.demanda > instance.Q:
                continue

            route = [depot, node.id, depot]

            cost = (
                instance.cost_matrix[depot][node.id] +
                instance.cost_matrix[node.id][depot]
            )

            # opcional: checar tempo
            time = (
                instance.time_matrix[depot][node.id] +
                instance.time_matrix[node.id][depot]
            )

            if time > instance.Y:
                continue

            col = cls(route, cost, time)
            columns.append(col)
        
        print(f"Criada rota: {route} | custo={cost:.2f} | tempo={time:.2f}")

        return columns
