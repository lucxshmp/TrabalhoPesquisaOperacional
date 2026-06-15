"""
Gera um dataset MISTO (dados reais + clientes/pedidos sintéticos) para tornar a
instância do PRV geograficamente desafiadora — assim a ILS-RVND tem o que
otimizar (hoje os pedidos reais caem em só ~10 locais distintos).

O que faz (tudo reprodutível com seed fixa):
  1. Lê `dadosPO.xlsx` (NÃO altera o original) — preserva os nomes de coluna.
  2. Cria N clientes sintéticos espalhados por coordenadas distintas na região
     metropolitana de BH, com CEPs sintéticos de 8 dígitos (prefixo 9000xxxx).
  3. Registra as coordenadas desses CEPs no geocoding_cache.json do BCP de forma
     ADITIVA (faz backup .bak antes; as entradas reais não são tocadas) — assim
     o loader resolve as coords sem depender de rede, e o BCP usa o mesmo subset.
  4. Gera pedidos sintéticos (demanda=1) ao longo de junho/2026, mais em dias
     úteis, para que os cenários dia/semana/mês fiquem mais densos e espalhados.
  5. Escreve `dadosPO_real_mais_sintetico.xlsx` (real + sintético).

Uso:
    python gerar_dataset_sintetico.py
    python gerar_dataset_sintetico.py --n-clientes 80 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd

ILS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR = os.path.join(ILS_DIR, "datasets")
CACHE_PATH = os.path.join(ILS_DIR, "..", "BCP", "data", "geocoding_cache.json")

# Bounding box aproximada da região metropolitana de BH (espalha os locais).
LAT_MIN, LAT_MAX = -20.00, -19.80
LON_MIN, LON_MAX = -44.12, -43.85

# Faixas de IDs sintéticos — bem fora das faixas reais (80x/90x) p/ evitar colisão.
CLIENTE_ID_BASE = 900_000_000
PEDIDO_ID_BASE = 9_000_000_000
CEP_BASE = 90_000_000  # vira "9000xxxx" (8 dígitos) ao formatar


def gerar_clientes(n, rng):
    """Cria n clientes sintéticos com coords distintas e CEPs de 8 dígitos."""
    lats = rng.uniform(LAT_MIN, LAT_MAX, size=n)
    lons = rng.uniform(LON_MIN, LON_MAX, size=n)
    clientes, coords_por_cep = [], {}
    for i in range(n):
        cid = CLIENTE_ID_BASE + i + 1
        cep_num = CEP_BASE + i + 1           # 90000001, 90000002, ...
        cep_str = f"{cep_num:08d}"           # exatamente 8 dígitos
        clientes.append({
            "idcliente": cid,
            "nome concessionaria ": f"CLIENTE SINTETICO {i + 1:03d}",
            "cidade": "BELO HORIZONTE",
            "cep": cep_str,
        })
        coords_por_cep[cep_str] = [float(lats[i]), float(lons[i])]
    return clientes, coords_por_cep


def gerar_pedidos(clientes, rng, por_dia_util=35, por_fim_semana=12):
    """Pedidos sintéticos (demanda=1) ao longo de junho/2026."""
    ids = [c["idcliente"] for c in clientes]
    dias = pd.date_range("2026-06-01", "2026-06-30", freq="D")
    pedidos, contador = [], 0
    for dia in dias:
        n_dia = por_fim_semana if dia.weekday() >= 5 else por_dia_util
        for _ in range(n_dia):
            pedidos.append({
                "id pedido": PEDIDO_ID_BASE + contador,
                "id cliente": int(rng.choice(ids)),
                "demanda": 1,
                "data ": dia,
            })
            contador += 1
    return pedidos


def atualizar_cache(coords_por_cep):
    """Adiciona (somente novas chaves) as coords sintéticas ao cache do BCP."""
    cache = {}
    if os.path.exists(CACHE_PATH):
        shutil.copy2(CACHE_PATH, CACHE_PATH + ".bak")
        with open(CACHE_PATH) as f:
            cache = json.load(f)
    n_antes = len(cache)
    novas = 0
    for cep, coords in coords_por_cep.items():
        if cep not in cache:          # ADITIVO: nunca sobrescreve chave existente
            cache[cep] = coords
            novas += 1
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"  cache: {n_antes} entradas -> {len(cache)} (+{novas} sintéticas; "
          f"backup em {os.path.basename(CACHE_PATH)}.bak)")


def main():
    ap = argparse.ArgumentParser(description="Gera dataset misto real+sintético")
    ap.add_argument("--n-clientes", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--por-dia-util", type=int, default=35)
    ap.add_argument("--por-fim-semana", type=int, default=12)
    ap.add_argument("--entrada", default=os.path.join(DATASETS_DIR, "dadosPO.xlsx"))
    ap.add_argument("--saida",
                    default=os.path.join(DATASETS_DIR, "dadosPO_real_mais_sintetico.xlsx"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # 1. lê o real SEM alterar (preserva nomes de coluna originais)
    xl = pd.ExcelFile(args.entrada)
    conc_real = xl.parse("concessionárias")
    ped_real = xl.parse("pedidos")
    print(f"real: {len(conc_real)} concessionárias, {len(ped_real)} pedidos")

    # 2-3. clientes sintéticos + coords no cache
    clientes, coords_por_cep = gerar_clientes(args.n_clientes, rng)
    atualizar_cache(coords_por_cep)

    # 4. pedidos sintéticos
    pedidos = gerar_pedidos(clientes, rng,
                            por_dia_util=args.por_dia_util,
                            por_fim_semana=args.por_fim_semana)
    print(f"sintético: {len(clientes)} clientes, {len(pedidos)} pedidos")

    # 5. concatena (real + sintético) mantendo as colunas do real
    conc_out = pd.concat([conc_real, pd.DataFrame(clientes)], ignore_index=True)
    ped_out = pd.concat([ped_real, pd.DataFrame(pedidos)], ignore_index=True)
    conc_out = conc_out[list(conc_real.columns)]
    ped_out = ped_out[list(ped_real.columns)]

    with pd.ExcelWriter(args.saida, engine="openpyxl") as w:
        conc_out.to_excel(w, sheet_name="concessionárias", index=False)
        ped_out.to_excel(w, sheet_name="pedidos", index=False)
    print(f"misto: {len(conc_out)} concessionárias, {len(ped_out)} pedidos")
    print(f"-> {args.saida}")


if __name__ == "__main__":
    main()
