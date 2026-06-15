import os
import sys

# Garante acesso ao geocoding e estruturas do BCP
BCP_DIR = os.path.join(os.path.dirname(__file__), "..", "BCP")
sys.path.insert(0, os.path.abspath(BCP_DIR))

import pandas as pd
from data.instance_reader import obter_coordenadas_por_cep, haversine
from data.structures import Node, Instance


def load_instance(path, depot_lat, depot_lon, Q, Y, K, max_clientes=None,
                  data_inicio=None, data_fim=None):
    """
    Lê dadosPO.xlsx (abas 'concessionárias' e 'pedidos') e constrói a Instance.

    Colunas esperadas:
      concessionárias: idcliente | nome concessionaria  | cidade | cep
      pedidos:         id pedido | id cliente | demanda | data

    max_clientes: se informado, limita a instância aos primeiros N pedidos
                  geocodificáveis (depósito não conta). Útil para reduzir a
                  escala (ex.: 300, a do artigo) e o tempo do ILS.
    data_inicio / data_fim: filtram os pedidos por período (inclusive), no
                  formato 'YYYY-MM-DD'. Usados pelos cenários (dia/semana/mês).
    """
    xl = pd.ExcelFile(path)
    df_conc = xl.parse("concessionárias")
    df_ped  = xl.parse("pedidos")

    # normaliza nomes de colunas (remove espaços extras)
    df_conc.columns = df_conc.columns.str.strip()
    df_ped.columns  = df_ped.columns.str.strip()

    # renomeia para nomes internos padronizados
    df_conc = df_conc.rename(columns={
        "idcliente":            "id",
        "nome concessionaria":  "nome",
    })
    df_ped = df_ped.rename(columns={
        "id pedido":  "id",
        "id cliente": "concessionaria_id",
    })

    # filtra pedidos por período (cenários dia/semana/mês), inclusive
    if data_inicio is not None or data_fim is not None:
        datas = pd.to_datetime(df_ped["data"], errors="coerce", dayfirst=True).dt.normalize()
        mask = pd.Series(True, index=df_ped.index)
        if data_inicio is not None:
            mask &= datas >= pd.Timestamp(data_inicio).normalize()
        if data_fim is not None:
            mask &= datas <= pd.Timestamp(data_fim).normalize()
        df_ped = df_ped[mask].copy()

    # filtra apenas concessionárias que têm pedidos
    ids_com_pedido = df_ped["concessionaria_id"].unique()
    df_conc = df_conc[df_conc["id"].isin(ids_com_pedido)].copy()

    # geocodifica concessionárias (cache em BCP/data/geocoding_cache.json)
    coordenadas = {}
    for _, row in df_conc.iterrows():
        coords = obter_coordenadas_por_cep(row["cep"])
        if coords is None:
            print(f"  [aviso] CEP {row['cep']} sem coordenadas — cliente {row['id']} ignorado")
        else:
            coordenadas[row["id"]] = coords

    # constrói lista de nós: nó 0 = depósito, nós 1..n = um por pedido
    nodes = [Node(id=0, latitude=depot_lat, longitude=depot_lon, demanda=0)]

    i = 1
    for _, ped in df_ped.iterrows():
        if max_clientes is not None and i > max_clientes:
            break
        cid = ped["concessionaria_id"]
        if cid not in coordenadas:
            continue
        lat, lon = coordenadas[cid]
        nodes.append(Node(
            id=i,
            latitude=lat,
            longitude=lon,
            demanda=int(ped["demanda"]),
            pedido_id=int(ped["id"]),
        ))
        i += 1

    # matrizes de custo e tempo
    n = len(nodes)
    cost_matrix = [[0.0] * n for _ in range(n)]
    time_matrix = [[0.0] * n for _ in range(n)]

    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            dist = haversine(
                nodes[a].latitude, nodes[a].longitude,
                nodes[b].latitude, nodes[b].longitude,
            )
            cost_matrix[a][b] = dist
            time_matrix[a][b] = dist / 40.0  # horas (velocidade média 40 km/h)

    return Instance(
        nodes=nodes,
        cost_matrix=cost_matrix,
        time_matrix=time_matrix,
        Q=Q,
        Y=Y,
        K=K,
    )
