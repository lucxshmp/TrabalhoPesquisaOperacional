import requests
from geopy.geocoders import Nominatim


class Concessionaria: 
    def __init__(self, id, nome, cep):
        self.id = id
        self.nome = nome
        self.cep = cep    
        self.latitude = None
        self.longitude = None

class Pedido:
    def __init__(self, id, concessionaria_id, demanda):
        self.id = id
        self.concessionaria_id = concessionaria_id
        self.demanda = demanda


class Node:
    def __init__(self, id, latitude, longitude, demanda, pedido_id=None):
        self.id = id
        self.latitude = latitude
        self.longitude = longitude
        self.demanda = demanda
        self.pedido_id = pedido_id


class Instance:
    def __init__(self, nodes, cost_matrix, time_matrix, Q, Y, K):
        self.nodes = nodes
        self.cost_matrix = cost_matrix
        self.time_matrix = time_matrix
        self.Q = Q
        self.Y = Y
        self.K = K

# Uso
# c = Concessionaria(1, "concessionaria lucas", "31310-490")
# resultado = c.obter_coordenadas_por_cep()

# print(resultado)
# print(c.latitude, c.longitude) 
