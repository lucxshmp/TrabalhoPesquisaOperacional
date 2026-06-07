from config import CONCESSIONARIAS_PATH, PEDIDOS_PATH
from data.instance_reader import load_instance
from master.columns import Column
from master.master_model import MasterModel
from pricing.pricing import pricing
from gurobipy import GRB

def main():

    # ----------------------
    # carregar instance
    # ----------------------
    instance = load_instance(
        CONCESSIONARIAS_PATH,
        PEDIDOS_PATH,
        depot_lat=-19.9,
        depot_lon=-44.1,
        Q=100,
        Y=8,
        K=20
    )

    print("Nodes:", len(instance.nodes))

    # ----------------------
    # colunas iniciais
    # ----------------------
    columns = Column.create_initial_routes(instance)

    print("Colunas iniciais:", len(columns))

    iteration = 0

    # ----------------------
    # LOOP DE COLUMN GENERATION
    # ----------------------
    while True:

        print(f"\n=== ITERAÇÃO {iteration} ===")

        master = MasterModel(instance, columns)
        master.build_model()
        master.solve()

        if master.model.status != 2:
            print("Modelo inviável!")
            break

        master.print_solution()

        # ----------------------
        # pegar duais
        # ----------------------
        duals = master.get_duals()

        print("Duais:", duals)

        # ----------------------
        # pricing
        # ----------------------
        new_columns = pricing(instance, duals)

        # ----------------------
        # critério de parada
        # ----------------------
        if not new_columns:
            print("Nenhuma nova coluna encontrada. FIM.")
            break

        print(f"{len(new_columns)} novas colunas adicionadas")

        # adiciona no conjunto
        columns.extend(new_columns)

        iteration += 1

    
    print("\n=== RESOLVENDO MODELO FINAL INTEIRO ===")

    # mudar variáveis pra binário
    for var in master.lambda_vars.values():
        var.vtype = GRB.BINARY

    master.model.optimize()

    master.print_solution()


if __name__ == "__main__":
    main()