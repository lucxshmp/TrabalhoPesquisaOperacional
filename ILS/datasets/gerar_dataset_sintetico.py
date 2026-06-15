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


def gerar_clientes(n, rng, offset=0):
    """Cria n clientes sintéticos com coords distintas e CEPs de 8 dígitos.

    offset desloca os IDs/CEPs para um bloco disjunto — assim datasets com
    quantidades diferentes de clientes não compartilham chaves no cache.
    """
    lats = rng.uniform(LAT_MIN, LAT_MAX, size=n)
    lons = rng.uniform(LON_MIN, LON_MAX, size=n)
    clientes, coords_por_cep = [], {}
    for i in range(n):
        cid = CLIENTE_ID_BASE + offset + i + 1
        cep_num = CEP_BASE + offset + i + 1   # 8 dígitos (offset mantém disjunto)
        cep_str = f"{cep_num:08d}"
        clientes.append({
            "idcliente": cid,
            "nome concessionaria ": f"CLIENTE SINTETICO {offset + i + 1:04d}",
            "cidade": "BELO HORIZONTE",
            "cep": cep_str,
        })
        coords_por_cep[cep_str] = [float(lats[i]), float(lons[i])]
    return clientes, coords_por_cep


def gerar_pedidos(clientes, rng, por_dia_util=35, por_fim_semana=12, offset=0):
    """Pedidos sintéticos (demanda=1) ao longo de junho/2026."""
    ids = [c["idcliente"] for c in clientes]
    dias = pd.date_range("2026-06-01", "2026-06-30", freq="D")
    pedidos, contador = [], 0
    for dia in dias:
        n_dia = por_fim_semana if dia.weekday() >= 5 else por_dia_util
        for _ in range(n_dia):
            pedidos.append({
                "id pedido": PEDIDO_ID_BASE + offset * 1_000_000 + contador,
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


def gerar_dataset(n_clientes, saida, entrada=None, seed=42, offset=0,
                  por_dia_util=35, por_fim_semana=12, verbose=True):
    """Gera um dataset misto (real + n_clientes sintéticos) e retorna o caminho.

    offset reserva um bloco disjunto de IDs/CEPs — use valores diferentes para
    datasets distintos (ex.: o experimento que varia o nº de clientes) p/ que as
    coordenadas no cache não se misturem entre eles.
    """
    entrada = entrada or os.path.join(DATASETS_DIR, "dadosPO.xlsx")
    rng = np.random.default_rng(seed)

    # 1. lê o real SEM alterar (preserva nomes de coluna originais)
    xl = pd.ExcelFile(entrada)
    conc_real = xl.parse("concessionárias")
    ped_real = xl.parse("pedidos")

    # 2-3. clientes sintéticos + coords no cache (aditivo, com backup)
    clientes, coords_por_cep = gerar_clientes(n_clientes, rng, offset=offset)
    atualizar_cache(coords_por_cep)

    # 4. pedidos sintéticos
    pedidos = gerar_pedidos(clientes, rng, por_dia_util=por_dia_util,
                            por_fim_semana=por_fim_semana, offset=offset)

    # 5. concatena (real + sintético) mantendo as colunas do real
    conc_out = pd.concat([conc_real, pd.DataFrame(clientes)], ignore_index=True)
    ped_out = pd.concat([ped_real, pd.DataFrame(pedidos)], ignore_index=True)
    conc_out = conc_out[list(conc_real.columns)]
    ped_out = ped_out[list(ped_real.columns)]

    with pd.ExcelWriter(saida, engine="openpyxl") as w:
        conc_out.to_excel(w, sheet_name="concessionárias", index=False)
        ped_out.to_excel(w, sheet_name="pedidos", index=False)

    if verbose:
        print(f"real: {len(conc_real)} concessionárias, {len(ped_real)} pedidos | "
              f"sintético: {len(clientes)} clientes, {len(pedidos)} pedidos")
        print(f"misto: {len(conc_out)} concessionárias, {len(ped_out)} pedidos "
              f"-> {saida}")
    return saida


def main():
    ap = argparse.ArgumentParser(description="Gera dataset misto real+sintético")
    ap.add_argument("--n-clientes", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--por-dia-util", type=int, default=35)
    ap.add_argument("--por-fim-semana", type=int, default=12)
    ap.add_argument("--entrada", default=os.path.join(DATASETS_DIR, "dadosPO.xlsx"))
    ap.add_argument("--saida",
                    default=os.path.join(DATASETS_DIR, "dadosPO_real_mais_sintetico.xlsx"))
    args = ap.parse_args()
    gerar_dataset(args.n_clientes, args.saida, entrada=args.entrada,
                  seed=args.seed, offset=args.offset,
                  por_dia_util=args.por_dia_util, por_fim_semana=args.por_fim_semana)


if __name__ == "__main__":
    main()
