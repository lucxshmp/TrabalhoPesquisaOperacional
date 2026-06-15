#Implementar funções de leitura das exportações do sistema da Stellantis
import json
import math
import os
import requests
from geopy.geocoders import Nominatim
import pandas as pd
from config import CONCESSIONARIAS_PATH, PEDIDOS_PATH
from data.structures import Concessionaria, Pedido, Node, Instance
import time
import os

#  1. obter lat. e long a partir do cep

def buscar_coordenada_web_fallback(cep_limpo):
    """
    Busca o CEP usando o ViaCEP (estável) + Nominatim OpenStreetMap de forma limpa.
    """
    # 1. Busca o endereço no ViaCEP
    url_viacep = f"https://viacep.com.br/ws/{cep_limpo}/json/"
    try:
        response = requests.get(url_viacep, timeout=3)
        if response.status_code == 200:
            dados = response.json()
            if "erro" in dados:
                return None
            
            logradouro = dados.get('logradouro', '')
            cidade = dados.get('localidade', '')
            uf = dados.get('uf', '')
            
            if cidade and uf:
                # 2. Transforma o endereço em Lat/Lon
                query = f"{logradouro}, {cidade} - {uf}, Brasil".replace(" ", "+")
                url_osm = f"https://nominatim.openstreetmap.org/search?q={query}&format=json&limit=1"
                headers = {'User-Agent': 'Stellantis_Logistics_App_Contact_lucas@domain.com'}
                
                res_osm = requests.get(url_osm, headers=headers, timeout=3)
                if res_osm.status_code == 200:
                    dados_osm = res_osm.json()
                    if dados_osm:
                        return float(dados_osm[0].get('lat')), float(dados_osm[0].get('lon'))
    except Exception:
        pass
    return None

def preencher_coordenadas_planilha(path_concessionarias):
    cache_path = "base_ceps_logradouros_nacional.csv"
    
    print("Lendo sua planilha de clientes Stellantis (11.338 linhas)...")
    df_clientes = pd.read_excel(path_concessionarias)
    
    if 'latitude' not in df_clientes.columns:
        df_clientes['latitude'] = None
    if 'longitude' not in df_clientes.columns:
        df_clientes['longitude'] = None
        
    # Limpa e padroniza o CEP para strings de 8 dígitos exatos
    df_clientes['cep_limpo'] = df_clientes['cep'].astype(str).str.replace(r'\D', '', regex=True).str.zfill(8)

    vazios_antes = df_clientes['latitude'].isna().sum()
    print(f"Clientes precisando de coordenadas: {vazios_antes}")

    if vazios_antes == 0:
        print("Todas as linhas já estão preenchidas")
        df_clientes = df_clientes.drop(columns=['cep_limpo'], errors='ignore')
        return

    # -------------------------------------------------------------------------
    # DOWNLOAD DA BASE REAL POR LOGRADOURO (Feito apenas uma vez)
    # -------------------------------------------------------------------------
    if not os.path.exists(cache_path):
        print("\n Baixando base nacional unificada com precisão por RUA...")
        print("Aguarde cerca de 30 segundos (é um arquivo compactado leve)...")
        
        # Link estável de dump de alta performance (com coordenadas por logradouro real)
        url_dados = "https://github.com/mtheis/ceps-brasil/raw/master/dados/ceps.csv.gz"
        
        try:
            # Baixa e descompacta em memória instantaneamente
            df_base = pd.read_csv(url_dados, compression='gzip', sep=';', dtype={'cep': str})
            df_base = df_base[['cep', 'latitude', 'longitude']].dropna()
            df_base['cep'] = df_base['cep'].str.replace(r'\D', '', regex=True).str.zfill(8)
            df_base = df_base.drop_duplicates(subset=['cep'])
            
            # Salva o cache na sua máquina
            df_base.to_csv(cache_path, index=False)
            print("Base salva localmente. As próximas rodadas serão instantâneas!")
        except Exception as e:
            print(f"Erro no download: {e}. Verifique sua conexão de rede.")
            return
    else:
        print("Carregando banco de CEPs do cache local...")
        df_base = pd.read_csv(cache_path, dtype={'cep': str})

    # -------------------------------------------------------------------------
    # CRUZAMENTO SEM REPETIÇÃO (Velocidade de fração de segundo)
    # -------------------------------------------------------------------------

    print("Processando mapeamento em memória...")
    dict_lat = dict(zip(df_base['cep'], df_base['latitude']))
    dict_lon = dict(zip(df_base['cep'], df_base['longitude']))

    vazios_mask = df_clientes['latitude'].isna() | df_clientes['longitude'].isna()
    
    # Preenche de forma cirúrgica mantendo o que você já tinha feito na mão
    df_clientes.loc[vazios_mask, 'latitude'] = df_clientes.loc[vazios_mask, 'cep_limpo'].map(dict_lat)
    df_clientes.loc[vazios_mask, 'longitude'] = df_clientes.loc[vazios_mask, 'cep_limpo'].map(dict_lon)

    # -------------------------------------------------------------------------
    # SALVAMENTO NA PLANILHA ORIGINAL
    # -------------------------------------------------------------------------
    df_clientes = df_clientes.drop(columns=['cep_limpo'])
    df_clientes.to_excel(path_concessionarias, index=False)
    
    vazios_depois = df_clientes['latitude'].isna().sum()
    print(f"Novos clientes localizados: {vazios_antes - vazios_depois}")
    print(f"Clientes não encontrados (CEPs inválidos ou muito recentes): {vazios_depois}")

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
    concessionarias_invalidas = set()

    for _, row in df.iterrows():

        c = Concessionaria(
            id=row['idcliente'],
            nome=row['nome concessionaria'],
            cep=row['cep']
        )

        lat = row.get('latitude')
        lon = row.get('longitude')

        if pd.notna(lat) and pd.notna(lon):
            c.latitude = float(lat)
            c.longitude = float(lon)

        else:
            coords = obter_coordenadas_por_cep(c.cep)

            if coords is None:
                print(f"Erro ao obter coordenadas para CEP {c.cep}")
                concessionarias_invalidas.add(c.id)
                continue

            c.latitude, c.longitude = coords

        concessionarias[c.id] = c

    return concessionarias, concessionarias_invalidas

def load_pedidos(path):
    df = pd.read_excel(path)

    pedidos = []

    for _, row in df.iterrows():
        p = Pedido(
            id=row['id pedido'],
            concessionaria_id=row['id cliente'],
            demanda=row.get('demanda')
        )
        pedidos.append(p)

    return pedidos


# 4. criar nós para cada pedido

def construir_nodes(pedidos, concessionarias, depot_lat, depot_lon):
    nodes = []

    # Nó 0: Depósito
    nodes.append(Node(
        id=0,
        latitude=depot_lat,
        longitude=depot_lon,
        demanda=0,
        pedido_id=None
    ))

    # Agrupa a demanda por concessionária única
    demanda_por_concessionaria = {}
    for pedido in pedidos:
        c_id = pedido.concessionaria_id
        if c_id not in demanda_por_concessionaria:
            demanda_por_concessionaria[c_id] = 0
        demanda_por_concessionaria[c_id] += pedido.demanda

    # Cria os nós garantindo sequência perfeita: 1, 2, 3, 4... sem buracos!
    i = 1
    for c_id, demanda_total in demanda_por_concessionaria.items():
        c = concessionarias[c_id]

        node = Node(
            id=i,
            latitude=c.latitude,
            longitude=c.longitude,
            demanda=demanda_total,
            pedido_id=c_id
        )
        nodes.append(node)
        i += 1  # Esse contador só sobe quando o nó REALMENTE é criado

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
    concessionarias, invalidas = load_concessionarias(path_concessionarias)
    pedidos = load_pedidos(path_pedidos)
    total_pedidos = len(pedidos)

    pedidos = [
        p for p in pedidos 
        if p.concessionaria_id not in invalidas
    ]

    print(
    f"Pedidos removidos por CEP inválido: "
    f"{total_pedidos - len(pedidos)}")

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

