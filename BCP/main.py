from config import CONCESSIONARIAS_PATH, PEDIDOS_PATH
from data.instance_reader import load_instance
from master.columns import Column
from master.master_model import MasterModel
from pricing.pricing import pricing
from gurobipy import GRB


def run_model(conc_path, ped_path, Q, Y, K):

    instance = load_instance(
        conc_path,
        ped_path,
        depot_lat=-19.9,
        depot_lon=-44.1,
        Q=Q,
        Y=Y,
        K=K
    )

    columns = Column.create_initial_routes(instance)
    master = MasterModel(instance, columns)
    master.build_model()

    iteration = 0

    # =========================
    # COLUMN GENERATION
    # =========================
    while True:
        master.solve()

        if master.model.Status != 2:
            break

        duals = master.get_duals()
        new_columns = pricing(instance, duals)

        if not new_columns:
            break

        for col in new_columns:
            columns.append(col)
            r = len(columns) - 1

            new_var = master.model.addVar(
                vtype=GRB.CONTINUOUS,
                lb=0.0,
                ub=1.0,
                obj=col.cost,
                name=f"lambda_{r}"
            )

            master.lambda_vars[r] = new_var

            for node_id in col.route:
                if node_id == 0:
                    continue
                constr = master.model.getConstrByName(f"cover_{node_id}")
                if constr:
                    master.model.chgCoeff(constr, new_var, 1.0)

            vehicle_constr = master.model.getConstrByName("vehicle_limit")
            if vehicle_constr:
                master.model.chgCoeff(vehicle_constr, new_var, 1.0)

        master.model.update()
        iteration += 1

    # =========================
    # FASE INTEIRA (FINAL)
    # =========================
    custo = None
    rotas_escolhidas = []

    if master.model.Status == 2:

        # converter variáveis para binárias
        for var in master.lambda_vars.values():
            var.vtype = GRB.BINARY

        # remover folgas (Big-M)
        for slack_var in master.slack_vars.values():
            slack_var.ub = 0.0

        master.model.update()

        master.model.optimize()

        if master.model.Status == GRB.OPTIMAL:

            custo = master.model.ObjVal

            for r, var in master.lambda_vars.items():
                if var.X > 0.5:
                    rota = columns[r].route
                    rotas_escolhidas.append(rota)

    # =========================
    # RETORNO FINAL
    # =========================
    return {
        "colunas": len(columns),
        "iteracoes": iteration,
        "status": master.model.Status,
        "custo": custo,
        "rotas": rotas_escolhidas
    }


def main():

    # ----------------------
    # Carregar instancia
    # ----------------------
    instance = load_instance(
        CONCESSIONARIAS_PATH,
        PEDIDOS_PATH,
        depot_lat=-19.9,
        depot_lon=-44.1,
        Q=500,
        Y=48,
        K=10
    )

    print("Nodes:", len(instance.nodes))

    # ----------------------
    # Criar colunas iniciais (Garante viabilidade de todos os clientes)
    # ----------------------
    columns = Column.create_initial_routes(instance)
    print("Colunas iniciais:", len(columns))

    # ----------------------
    # Inicializar o Modelo Mestre (Criado uma unica vez antes do loop)
    # ----------------------
    master = MasterModel(instance, columns)
    master.build_model()

    iteration = 0

    # ----------------------
    # LOOP DE COLUMN GENERATION
    # ----------------------
    while True:

        print(f"\n=== ITERACAO {iteration} ===")

        # Resolve a relaxacao linear atual do mestre
        master.solve()

        if master.model.Status != 2: 
            print("Modelo inviavel na relaxacao linear!")
            break

        master.print_solution()

        # ----------------------
        # Coletar precos duais estruturados
        # ----------------------
        duals = master.get_duals()
        print("Duais obtidos para precificacao.")

        # ----------------------
        # Chamar subproblema de precificacao (Algoritmo de Pareto)
        # ----------------------
        new_columns = pricing(instance, duals)

        # ----------------------
        # Criterio de parada: Fim do Column Generation
        # ----------------------
        if not new_columns:
            print("\nNenhuma nova coluna encontrada com custo reduzido negativo. FIM DO LOOP.")
            break

        print(f"{len(new_columns)} novas colunas encontradas pelo pricing.")

        # ----------------------
        # INJECAO DINAMICA: Adicionar as novas colunas direto no modelo Gurobi existente
        # ----------------------
        for col in new_columns:
            # 1. Armazena no historico de colunas locais
            columns.append(col)
            r = len(columns) - 1  # Novo indice da coluna
            
            # 2. Cria a nova variavel lambda de forma continua [0, 1] no Gurobi
            new_var = master.model.addVar(
                vtype=GRB.CONTINUOUS,
                lb=0.0,
                ub=1.0,
                obj=col.cost,  # Define o coeficiente de custo direto na Funcao Objetivo
                name=f"lambda_{r}"
            )
            
            # Guarda a referencia na estrutura interna da classe
            master.lambda_vars[r] = new_var
            
            # 3. Adiciona os coeficientes nas restricoes de cobertura de clientes vigentes
            for node_id in col.route:
                if node_id == 0:
                    continue  # ignora deposito nas restricoes de cobertura
                
                # Busca a restricao correspondente no mestre por nome
                constr = master.model.getConstrByName(f"cover_{node_id}")
                if constr is not None:
                    # Injeta o coeficiente de cobertura (coef=1) para esta nova variavel
                    master.model.chgCoeff(constr, new_var, 1.0)
            
            # 4. Adiciona o coeficiente na restricao de limite global de veiculos
            vehicle_constr = master.model.getConstrByName("vehicle_limit")
            if vehicle_constr is not None:
                master.model.chgCoeff(vehicle_constr, new_var, 1.0)

        # Atualiza a estrutura interna do Gurobi apos a insercao das colunas
        master.model.update()
        iteration += 1

    # -------------------------------------------------------------------------
    # RESOLVENDO MODELO FINAL INTEIRO (Heuristica Price-and-Branch)
    # -------------------------------------------------------------------------
    if master and master.model.Status == 2:
        print("\n=== RESOLVENDO MODELO FINAL INTEIRO ===")
        print(f"Total de colunas salvas no pool final integrado: {len(columns)}")

        # Converte as variaveis lambda do pool acumulado em binarias
        for var in master.lambda_vars.values():
            var.vtype = GRB.BINARY

        # BLOQUEIO DO BIG M: Zera o limite superior das folgas para obrigar o atendimento real
        for slack_var in master.slack_vars.values():
            slack_var.ub = 0.0

        master.model.update() 
        master.model.setParam('OutputFlag', 1)

        # Otimiza o problema IP final
        master.model.optimize()
        master.print_solution()
    else:
        print("\nNao foi possivel gerar o modelo inteiro porque a relaxacao linear falhou.")


if __name__ == "__main__":
    main()