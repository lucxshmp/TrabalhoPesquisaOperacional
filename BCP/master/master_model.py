import gurobipy as gp
from gurobipy import GRB


class MasterModel:

    def __init__(self, instance, columns):
        self.instance = instance
        self.columns = columns
        self.model = gp.Model("VRP_Master")
        self.lambda_vars = {}

    # --------------------------
    # construir modelo
    # --------------------------
    def build_model(self):

        # Variáveis λ_r
        for r, col in enumerate(self.columns):
            self.lambda_vars[r] = self.model.addVar(                
            vtype=GRB.CONTINUOUS,
                lb=0,
                ub=1,
                name=f"lambda_{r}"
            )


        self.model.update()

        # --------------------------
        # função objetivo
        # --------------------------
        self.model.setObjective(
            gp.quicksum(
                col.cost * self.lambda_vars[r]
                for r, col in enumerate(self.columns)
            ),
            GRB.MINIMIZE
        )

        # --------------------------
        # cada cliente atendido 1 vez
        # --------------------------
        for node in self.instance.nodes:

            if node.id == 0:
                continue

            self.model.addConstr(
                gp.quicksum(
                    self.lambda_vars[r]
                    for r, col in enumerate(self.columns)
                    if node.id in col.coverage
                ) == 1,
                name=f"cover_{node.id}"
            )

        # --------------------------
        # limite de veículos
        # --------------------------
        self.model.addConstr(
            gp.quicksum(self.lambda_vars[r] for r in self.lambda_vars)
            <= self.instance.K,
            name="vehicle_limit"
        )

    # --------------------------
    def solve(self):
        self.model.optimize()

    # --------------------------
    def print_solution(self):

        print("\n=== SOLUÇÃO ===")

        for r, var in self.lambda_vars.items():
            if var.X > 0.5:
                col = self.columns[r]
                print(f"Rota escolhida: {col.route} | custo={col.cost:.2f}")

        print("Custo total:", self.model.ObjVal)

    def get_duals(self):
        duals = {}

        for constr in self.model.getConstrs():
            if "cover_" in constr.ConstrName:
                node_id = int(constr.ConstrName.split("_")[1])
                duals[node_id] = constr.Pi

        return duals