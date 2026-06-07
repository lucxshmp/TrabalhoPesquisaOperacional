#Implementar funções de leitura das exportações do sistema da Stellantis 
import math
import requests
from geopy.geocoders import Nominatim
import pandas as pd

from config import CONCESSIONARIAS_PATH, PEDIDOS_PATH
from data.structures import Concessionaria, Pedido, Node, Instance


#  1. obter lat. e long a partir do cep


def obter_coordenadas_por_cep(cep):
    cep_limpo = ''.join(filter(str.isdigit, str(cep)))

    if len(cep_limpo) != 8:
        return None

    url_brasil_api = f"https://brasilapi.com.br/api/cep/v1/{cep_limpo}"

    try:
        response = requests.get(url_brasil_api, timeout=5)

        if response.status_code != 200:
            return None

        dados_endereco = response.json()

        endereco_completo = (
            f"{dados_endereco.get('street','')}, "
            f"{dados_endereco.get('neighborhood','')}, "
            f"{dados_endereco.get('city','')} - "
            f"{dados_endereco.get('state','')}, Brasil"
        )

        geolocator = Nominatim(user_agent="meu_app", timeout=10)
        localizacao = geolocator.geocode(endereco_completo)

        if localizacao:
            return localizacao.latitude, localizacao.longitude

    except Exception:
        return None

    return None



# 2. obter distância entre dois pontos considerando a curvatura da terra (harversine)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # raio da terra em km

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    return R * c  # distância em km


# 3. leitura das planilhas

def load_concessionarias(path):
    df = pd.read_excel(path)

    concessionarias = {}

    for _, row in df.iterrows():
        c = Concessionaria(
            id=row['id'],
            nome=row['concessionaria'],
            cep=row['cep']
        )

        coords = obter_coordenadas_por_cep(c.cep)

        if coords is None:
            raise Exception(f"Erro ao obter coordenadas para CEP {c.cep}")

        c.latitude, c.longitude = coords

        concessionarias[c.id] = c

    return concessionarias


def load_pedidos(path):
    df = pd.read_excel(path)

    pedidos = []

    for _, row in df.iterrows():
        p = Pedido(
            id=row['id'],
            concessionaria_id=row['concessionaria_id'],
            demanda=row.get('materiais', 1)
        )
        pedidos.append(p)

    return pedidos


# 4. criar nós para cada pedido

def construir_nodes(pedidos, concessionarias, depot_lat, depot_lon):
    nodes = []

    # depósito (id = 0)
    nodes.append(Node(
        id=0,
        latitude=depot_lat,
        longitude=depot_lon,
        demanda=0,
        pedido_id=None
    ))

    i = 1
    for pedido in pedidos:
        c = concessionarias[pedido.concessionaria_id]

        node = Node(
            id=i,
            latitude=c.latitude,
            longitude=c.longitude,
            demanda=pedido.demanda,
            pedido_id=pedido.id
        )

        nodes.append(node)
        i += 1

    return nodes


# 5. matriz de custo e tempo

def construir_matrizes(nodes, velocidade_media=40):
    n = len(nodes)

    cost_matrix = [[0]*n for _ in range(n)]
    time_matrix = [[0]*n for _ in range(n)]

    for i in range(n):
        for j in range(n):

            if i == j:
                cost_matrix[i][j] = 0
                time_matrix[i][j] = 0
                continue

            dist = haversine(
                nodes[i].latitude, nodes[i].longitude,
                nodes[j].latitude, nodes[j].longitude
            )

            cost_matrix[i][j] = dist
            time_matrix[i][j] = dist / velocidade_media  # horas

    return cost_matrix, time_matrix


# 6. criar instâncias

def load_instance(
    path_concessionarias,
    path_pedidos,
    depot_lat,
    depot_lon,
    Q,
    Y,
    K
):


    # 1. ler dados
    concessionarias = load_concessionarias(path_concessionarias)
    pedidos = load_pedidos(path_pedidos)

    # 2. construir nodes
    nodes = construir_nodes(pedidos, concessionarias, depot_lat, depot_lon)

    # 3. construir matrizes
    cost_matrix, time_matrix = construir_matrizes(nodes)

    # 4. criar instance
    instance = Instance(
        nodes=nodes,
        cost_matrix=cost_matrix,
        time_matrix=time_matrix,
        Q=Q,
        Y=Y,
        K=K
    )

    return instance

