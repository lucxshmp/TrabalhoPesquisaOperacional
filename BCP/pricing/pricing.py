import heapq
from master.columns import Column

class Label:
    """Representa o estado do veículo (vetor de recursos para a otimalidade de Pareto)."""
    def __init__(self, current_node, rc, demand, time, cost, parent_label=None):
        self.current_node = current_node
        self.rc = rc              # Custo reduzido acumulado
        self.demand = demand      # Recurso 1: Capacidade usada
        self.time = time          # Recurso 2: Tempo gasto
        self.cost = cost          # Custo real acumulado
        self.parent_label = parent_label

    def __lt__(self, other):
        """Comparar custos reduzidos para o Min-Heap."""
        return self.rc < other.rc
    
    def get_route(self):
        """Reconstrói o caminho percorrido pelo rótulo de trás para frente."""
        route = []
        curr = self
        while curr:
            route.append(curr.current_node)
            curr = curr.parent_label
        return route[::-1]


def pricing(instance, duals):
    """
    Subproblema de Precificação utilizando o Algoritmo de Rotulagem (Labeling)
    com base no critério de dominância de Pareto (Livro de Desaulniers).
    """
    MAX_NEW_COLUMNS = 40
    depot = 0
    num_nodes = len(instance.nodes)
    
    # Desempacota os duais estruturados vindos do MasterModel
    duais_clientes = duals["clientes"]
    dual_veiculo = duals["veiculo"]
    
    # Dicionário que armazena os vetores de recursos Pareto-ótimos para cada nó
    pareto_labels = {i: [] for i in range(num_nodes)}
    
    # Fila de Prioridades (Priority Queue) para processar os rótulos mais promissores primeiro
    pq = []
    
    # Inicializa o rótulo de partida no depósito
    initial_label = Label(current_node=depot, rc=0.0, demand=0, time=0.0, cost=0.0)
    heapq.heappush(pq, initial_label)
    pareto_labels[depot].append(initial_label)
    
    new_columns = []
    
    while pq:
        curr_label = heapq.heappop(pq)
        u = curr_label.current_node
        
        # Teste de dominância de Pareto: se o rótulo atual já foi dominado por outro no mesmo nó, descarta
        if any(other.rc <= curr_label.rc and other.demand <= curr_label.demand and other.time <= curr_label.time 
               for other in pareto_labels[u] if other != curr_label):
            continue
            
        # ---------------------------------------------------------------------
        # FECHAMENTO DA ROTA: Retorno ao Depósito
        # ---------------------------------------------------------------------
        if u == depot and curr_label.time > 0:
            # Aplica a equação exata de Dantzig-Wolfe deduzindo o dual global do veículo
            final_reduced_cost = curr_label.rc - dual_veiculo
            
            # Tolerância numérica padrão para identificar custos reduzidos negativos
            if final_reduced_cost < -1e-4:
                route = curr_label.get_route()
                col = Column(route, curr_label.cost, curr_label.time)
                
                # Evita duplicatas na mesma iteração
                if col not in new_columns:
                    new_columns.append(col)
                    print(f"Nova rota aglutinada promissora: {route} | rc={final_reduced_cost:.2f} | custo={curr_label.cost:.2f}")
                    
                    if len(new_columns) >= MAX_NEW_COLUMNS:
                        break
            continue

        # ---------------------------------------------------------------------
        # EXPANSÃO INTELIGENTE: Avança para as concessionárias vizinhas
        # ---------------------------------------------------------------------
        for neighbor in instance.nodes:
            v = neighbor.id
            
            # Evita loops estáticos ou transições inválidas no depósito
            if v == u or (v == depot and u == depot):
                continue

            # 1. Validação física dos Recursos (Capacidade e Janela de Tempo/Jornada)
            next_demand = curr_label.demand + neighbor.demanda
            if next_demand > instance.Q:
                continue
                
            next_time = curr_label.time + instance.time_matrix[u][v]
            if next_time > instance.Y:
                continue

            # Garante caminho elementar (sem repetir nenhuma concessionária na mesma viagem)
            current_route = curr_label.get_route()
            if v != depot and v in current_route:
                continue

            # 2. Atualização dos custos acumulados e do Custo Reduzido do arco
            next_cost = curr_label.cost + instance.cost_matrix[u][v]
            
            # O custo reduzido do arco deduz o preço dual do cliente de destino
            arc_rc = instance.cost_matrix[u][v] - (duais_clientes.get(v, 0) if v != depot else 0)
            next_rc = curr_label.rc + arc_rc

            # 3. FILTRO DE DOMINÂNCIA DE PARETO
            # Verifica se o novo caminho gerado já é pior (dominado) por algum vetor existente em 'v'
            dominated = False
            for other in pareto_labels[v]:
                if other.rc <= next_rc and other.demand <= next_demand and other.time <= next_time:
                    dominated = True
                    break
            if dominated:
                continue

            # Criou um novo estado Pareto-ótimo viável
            new_label = Label(v, next_rc, next_demand, next_time, next_cost, curr_label)

            # Remove da lista de 'v' rótulos antigos obsoletos (que foram dominados pelo novo)
            pareto_labels[v] = [
                other for other in pareto_labels[v]
                if not (new_label.rc <= other.rc and new_label.demand <= other.demand and new_label.time <= other.time)
            ]

            # Registra e empilha o novo rótulo para expansões futuras
            pareto_labels[v].append(new_label)
            heapq.heappush(pq, new_label)

    return new_columns