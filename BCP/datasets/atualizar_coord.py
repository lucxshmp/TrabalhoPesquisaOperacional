from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).parent

df = pd.read_excel(BASE_DIR / "concessionarias.xlsx")
coords = pd.read_csv(BASE_DIR / "coordenadas.csv")

# Planilha original
df = pd.read_excel("concessionarias.xlsx")

# Arquivo com as coordenadas
coords = pd.read_csv("coordenadas.csv")

# Mesmo tipo para evitar problemas no merge
df["idcliente"] = df["idcliente"].astype(str)
coords["idcliente"] = coords["idcliente"].astype(str)

# Remove latitude/longitude antigas caso existam
df = df.drop(columns=["latitude", "longitude"], errors="ignore")

# Junta os dados pelo idcliente
df = df.merge(
    coords[["idcliente", "latitude", "longitude"]],
    on="idcliente",
    how="left"
)

# Salva resultado
df.to_excel(
    "concessionarias_com_coordenadas.xlsx",
    index=False
)

print("Concluído!")