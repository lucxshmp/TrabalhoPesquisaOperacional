import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import math

# 1. Carregar os dados do CSV
# O arquivo deve conter as colunas: id, x, y, demanda
# Assumimos que o id=0 é o depósito inicial.
df = pd.read_csv('dados_dev.csv')

# Parâmetros gerais (Podem ser ajustados conforme sua necessidade)
Q = 50  # Capacidade máxima do caminhão
Y = 200 # Duração/Tempo máximo permitido para cada rota
num_caminhoes = 3
K = list(range(num_caminhoes)) # Conjunto de caminhões K

# Processamento dos Conjuntos
n = len(df) - 1 # Número de concessionárias (excluindo o depósito 0)
depot_row = df[df['id'] == 0].iloc[0]
# # O nó n+1 representa o retorno ao depósito (mesma coordenada e demanda 0)
df.loc[len(df)] = {'id': n + 1, 'Latitude': depot_row['Latitude'], 'Longitude': depot_row['Longitude'], 'Demanda': 0}

N = list(df['id']) # N = {0, ..., n, n+1}
C = [i for i in N if i != 0 and i != n + 1] # Conjunto de concessionárias C

# # Dicionários de coordenadas e demandas
coord = {row['id']: (row['Latitude'], row['Longitude']) for _, row in df.iterrows()}
d = {row['id']: row['Demanda'] for _, row in df.iterrows()}

# # Criando o Conjunto de Arcos (E) e Parâmetros (c_ij, t_ij)
E = []
c = {}
t = {}

for i in N:
    for j in N:
        # Condições do conjunto E: i != j, i != n+1 (não sai do fim), j != 0 (não volta pro início)
        if i != j and i != (n + 1) and j != 0:
            E.append((i, j))
            # Usando a distância euclidiana para custo e tempo
            dist = math.hypot(coord[i][0] - coord[j][0], coord[i][1] - coord[j][1])
            c[i, j] = dist
            t[i, j] = dist

# # 2. Inicializando o modelo Gurobi
modelo = gp.Model("VRP_Concessionarias")

# # 3. Variáveis de Decisão: x_ijk = 1 se o veículo k percorre o arco (i,j)
x = modelo.addVars(E, K, vtype=GRB.BINARY, name="x")

# # 4. Função Objetivo
modelo.setObjective(gp.quicksum(c[i, j] * x[i, j, k] for (i, j) in E for k in K), GRB.MINIMIZE)

# # 5. Restrições

# # 1º: Toda concessionária deve ser visitada exatamente 1 vez
for i in C:
    modelo.addConstr(
        gp.quicksum(x[i, j, k] for k in K for j in N if (i, j) in E) == 1,
        name=f"visita_{i}"
    )

# # 2º: Capacidade máxima do caminhão
for k in K:
    modelo.addConstr(
        gp.quicksum(d[i] * x[i, j, k] for (i, j) in E) <= Q,
        name=f"capacidade_{k}"
    )

# # 3º: Duração máxima da rota
for k in K:
    modelo.addConstr(
        gp.quicksum(t[i, j] * x[i, j, k] for (i, j) in E) <= Y,
        name=f"tempo_max_{k}"
    )

# # 4º: Conservação de fluxo
for h in C:
    for k in K:
        modelo.addConstr(
            gp.quicksum(x[i, h, k] for i in N if (i, h) in E) -
            gp.quicksum(x[h, j, k] for j in N if (h, j) in E) == 0,
            name=f"fluxo_{h}_{k}"
        )

# # 5º: Todo veículo k sai do depósito inicial (nó 0)
for k in K:
    modelo.addConstr(
        gp.quicksum(x[0, j, k] for j in N if (0, j) in E) == 1,
        name=f"sai_deposito_{k}"
    )

# # 6º: Todo veículo k chega no depósito final (nó n+1)
for k in K:
    modelo.addConstr(
        gp.quicksum(x[i, n+1, k] for i in N if (i, n+1) in E) == 1,
        name=f"chega_deposito_{k}"
    )

# # 6. Otimizar o modelo
modelo.optimize()

# # 7. Exibir os resultados (Lógica de exibição simplificada para ver a rota)
if modelo.Status == GRB.OPTIMAL:
    print(f"\nCusto Total (Distância): {modelo.ObjVal:.2f}")
    for k in K:
        print(f"\n--- Rota do caminhão {k} ---")
        for (i, j) in E:
            if x[i, j, k].X > 0.5:
                print(f" Arco {i} -> {j} | Custo: {c[i,j]:.2f}")
else:
    print(f"\nNenhuma solução ótima encontrada. Status do modelo: {modelo.Status}")