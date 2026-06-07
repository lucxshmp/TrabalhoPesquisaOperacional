import pandas as pd
import requests
import time
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

# 1. Configuração Inicial
# Lendo o arquivo 'dados.csv' separado por vírgula
print("Lendo o arquivo CSV...")
df = pd.read_csv('dados_dev.csv', sep=',')

# 2. Geocodificação (Descobrir Lat/Lon a partir do bairro)
geolocator = Nominatim(user_agent="gurobi_vrp_tester")

def obter_coordenadas(bairro):
    # Adicionamos "Belo Horizonte, Brasil" para ajudar a API a ser mais precisa
    query = f"{bairro}, Belo Horizonte, Brasil"
    try:
        location = geolocator.geocode(query, timeout=10)
        if location:
            return location.latitude, location.longitude
        else:
            print(f"Aviso: Não foi possível localizar o bairro {bairro}.")
            return None, None # Adicionado retorno None para caso não encontre
    
    except GeocoderTimedOut:
        print(f"Aviso: Tempo limite esgotado para o bairro {bairro}.")
        return None, None

latitudes = []
longitudes = []

print("Buscando coordenadas...")
for index, row in df.iterrows():
    lat, lon = obter_coordenadas(row['Bairro'])
    latitudes.append(lat)
    longitudes.append(lon)
    time.sleep(1) # Respeitando o limite de requisições gratuitas do OpenStreetMap
    print(f"{row['Bairro']}: Lat: {lat}, Lon: {lon}")

# 3. Inserindo as informações no DataFrame
df['Latitude'] = latitudes
df['Longitude'] = longitudes
df['Demanda'] = 1

# 4. Salvando o DataFrame atualizado em um novo arquivo CSV (ou sobrescrevendo o original)
print("Salvando as coordenadas no arquivo CSV...")
# O index=False evita que o pandas salve a coluna de numeração das linhas no CSV
df.to_csv('dados_dev.csv', sep=',', index=False, encoding='utf-8')

print("Processo concluído com sucesso! Arquivo salvo como 'dados_dev.csv'.")