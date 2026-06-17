import argparse
import numpy as np
import pandas as pd
import os

# Bounding box (BH)
LAT_MIN, LAT_MAX = -20.00, -19.80
LON_MIN, LON_MAX = -44.12, -43.85

CLIENTE_ID_BASE = 900_000_000
PEDIDO_ID_BASE = 9_000_000_000


def gerar_clientes(n, rng, offset=0):
    lats = rng.uniform(LAT_MIN, LAT_MAX, size=n)
    lons = rng.uniform(LON_MIN, LON_MAX, size=n)

    clientes = []
    for i in range(n):
        clientes.append({
            "idcliente": CLIENTE_ID_BASE + offset + i + 1,
            "nome concessionaria ": f"SINTETICO {i+1}",
            "cidade": "BELO HORIZONTE",
            "cep": f"{90000000 + offset + i + 1}",
            "latitude": float(lats[i]),
            "longitude": float(lons[i])
        })

    return clientes


def gerar_pedidos(clientes, rng, por_dia_util=30, por_fds=10, offset=0):
    ids = [c["idcliente"] for c in clientes]

    dias = pd.date_range("2026-06-01", "2026-06-30", freq="D")

    pedidos = []
    contador = 0

    for dia in dias:
        n_dia = por_fds if dia.weekday() >= 5 else por_dia_util

        for _ in range(n_dia):
            pedidos.append({
                "id pedido": PEDIDO_ID_BASE + offset * 1_000_000 + contador,
                "id cliente": int(rng.choice(ids)),
                "demanda": 1,
                "data ": dia
            })
            contador += 1

    return pedidos


def gerar_dataset(
    entrada_conc,
    entrada_ped,
    saida,
    n_clientes=80,
    seed=42,
    offset=0
):
    rng = np.random.default_rng(seed)

    # ✅ lê suas planilhas reais
    conc_real = pd.read_excel(entrada_conc, engine="openpyxl")
    ped_real = pd.read_excel(entrada_ped, engine="openpyxl")

    # ✅ gera sintético
    clientes_sint = gerar_clientes(n_clientes, rng, offset)
    pedidos_sint = gerar_pedidos(clientes_sint, rng, offset=offset)

    # ✅ concatena mantendo estrutura original
    conc_out = pd.concat([conc_real, pd.DataFrame(clientes_sint)], ignore_index=True)
    ped_out = pd.concat([ped_real, pd.DataFrame(pedidos_sint)], ignore_index=True)

    # ✅ garante ordem das colunas iguais ao original
    conc_out = conc_out[conc_real.columns]
    ped_out = ped_out[ped_real.columns]

    # ✅ salva
    with pd.ExcelWriter(saida, engine="openpyxl") as writer:
        conc_out.to_excel(writer, sheet_name="concessionarias", index=False)
        ped_out.to_excel(writer, sheet_name="pedidos", index=False)

    print("✅ Dataset gerado:")
    print(f"- Concessionarias: {len(conc_out)}")
    print(f"- Pedidos: {len(ped_out)}")
    print(f"Arquivo: {saida}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--conc", default="concessionarias_atualizada.xlsx")
    parser.add_argument("--ped", default="pedidos.xlsx")
    parser.add_argument("--saida", default="dataset_misto.xlsx")
    parser.add_argument("--n-clientes", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offset", type=int, default=0)

    args = parser.parse_args()

    gerar_dataset(
        args.conc,
        args.ped,
        args.saida,
        args.n_clientes,
        args.seed,
        args.offset
    )