class Column:
    def __init__(self, route, cost, time):
        self.route = route
        self.cost = cost
        self.coverage = set(route[1:-1])
        self.time = time

    def __eq__(self, other):
        """Define que duas colunas são iguais se tiverem a mesma rota exata."""
        if not isinstance(other, Column):
            return False
        return self.route == other.route

    def __hash__(self):
        """Permite usar a coluna em estruturas de dados como sets se necessário."""
        return hash(tuple(self.route))

    @staticmethod
    def create_initial_routes(instance):
        initial_columns = []
        depot = 0
        for node in instance.nodes:
            if node.id == depot:
                continue
            
            # Cria uma rota dedicada para garantir que o mestre comece viável
            route = [depot, node.id, depot]
            cost = instance.cost_matrix[depot][node.id] + instance.cost_matrix[node.id][depot]
            time = instance.time_matrix[depot][node.id] + instance.time_matrix[node.id][depot]
            
            # Cria o objeto de coluna
            initial_columns.append(Column(route, cost, time))
            
        return initial_columns