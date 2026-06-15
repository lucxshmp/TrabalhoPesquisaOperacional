import gurobipy as gp
from gurobipy import GRB

class MasterModel:

    def __init__(self, instance, columns):
        self.instance = instance
        self.columns = columns
        self.model = gp.Model("VRP_Master")
        self.model.setParam('OutputFlag', 0) 
        self.lambda_vars = {}
        self.slack_vars = {} # Armazena as variaveis Big M

    def build_model(self):
        # Um valor de M de 1000 a 5000 costuma ser ideal para evitar instabilidade numerica
        BIG_M = 2000.0 

        # Variaveis λ_r normais
        for r, col in enumerate(self.columns):
            self.lambda_vars[r] = self.model.addVar(                
                vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"lambda_{r}"
            )

        self.model.update()

        # ---------------------------------------------------------------------
        # RESTRICOES DE COBERTURA + VARIAVEIS ARTIFICIAIS (BIG M)
        # ---------------------------------------------------------------------
        for node in self.instance.nodes:
            if node.id == 0:
                continue

            # Cria a variavel de folga s_i para o cliente
            slack = self.model.addVar(
                vtype=GRB.CONTINUOUS, lb=0.0, name=f"slack_{node.id}"
            )
            self.slack_vars[node.id] = slack

            # Injeta a folga na restricao de cobertura (Set Covering >= 1)
            self.model.addConstr(
                gp.quicksum(
                    self.lambda_vars[r]
                    for r, col in enumerate(self.columns)
                    if node.id in col.route
                ) + slack >= 1.0,
                name=f"cover_{node.id}"
            )

        # Restricao de limite de veiculos
        self.model.addConstr(
            gp.quicksum(self.lambda_vars[r] for r in self.lambda_vars)
            <= self.instance.K,
            name="vehicle_limit"
        )

        self.model.update()

        # ---------------------------------------------------------------------
        # FUNCAO OBJETIVO (Custo das Rotas + Penalizacao das Folgas)
        # ---------------------------------------------------------------------
        self.model.setObjective(
            gp.quicksum(col.cost * self.lambda_vars[r] for r, col in enumerate(self.columns)) +
            gp.quicksum(BIG_M * self.slack_vars[node.id] for node in self.instance.nodes if node.id != 0),
            GRB.MINIMIZE
        )

    def solve(self):
        self.model.optimize()

    def print_solution(self):
        print("\n=== SOLUCAO MESTRE ===")
        for r, var in self.lambda_vars.items():
            if var.X > 0.01:
                col = self.columns[r]
                print(f"Rota escolhida (λ={var.X:.2f}): {col.route} | custo={col.cost:.2f}")
        
        # Monitoramento das variaveis Big M
        for node_id, var in self.slack_vars.items():
            if var.X > 0.01:
                print(f"Cliente {node_id} NAO atendido fisicamente (Usa folga s={var.X:.2f})")

        if self.model.Status == 2:
            print(f"Custo total da Funcao Objetivo (com penalizacoes): {self.model.ObjVal:.2f}")

    def get_duals(self):
        duals = {"clientes": {}, "veiculo": 0.0}
        for constr in self.model.getConstrs():
            if "cover_" in constr.ConstrName:
                node_id = int(constr.ConstrName.split("_")[1])
                duals["clientes"][node_id] = constr.Pi
            elif constr.ConstrName == "vehicle_limit":
                duals["veiculo"] = constr.Pi
        return duals