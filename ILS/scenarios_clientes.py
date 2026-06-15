"""
Cenários do ILS-RVND VARIANDO O NÚMERO DE CLIENTES (locais distintos).

Diferente de `scenarios.py` (que varia frota × período sobre os dados reais e é
mantido como está), aqui isolamos o **nº de clientes sintéticos** como a única
variável do experimento. Fixamos a janela (1 dia, 2026-06-05) e a frota (K=15);
só o espalhamento geográfico muda, evidenciando como a metaheurística escala e
quanto ela realmente otimiza conforme há mais locais distintos.

Para cada nível de clientes:
  - gera um dataset misto (real + N sintéticos) em datasets/, com bloco de
    IDs/CEPs disjunto (não altera dadosPO.xlsx; cache atualizado de forma aditiva);
  - roda o ILS-RVND e salva CSVs + 3 gráficos em
    resultados/cenarios_clientes/<N>clientes/.

Uso:
    python scenarios_clientes.py            # roda os níveis padrão (20/40/80)
    python scenarios_clientes.py --rapido   # params mínimos p/ validação rápida
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from datasets.gerar_dataset_sintetico import gerar_dataset
from instance_reader import load_instance
from scenarios import run_scenario

# Níveis de clientes a comparar (a variável do experimento)
COUNTS = [20, 40, 80]

# Dimensões FIXAS (isolam o nº de clientes)
DATA = "2026-06-05"      # 1 dia
K = 15                   # frota fixa


def _contar_locais(dados, data, Q, Y):
    """Nº de coordenadas distintas na janela (métrica-chave do experimento)."""
    inst = load_instance(dados, depot_lat=-19.9, depot_lon=-44.1, Q=Q, Y=Y, K=K,
                         data_inicio=data, data_fim=data)
    locs = {(round(nd.latitude, 6), round(nd.longitude, 6)) for nd in inst.nodes[1:]}
    return len(inst.nodes) - 1, len(locs)


def main():
    ap = argparse.ArgumentParser(description="Cenários ILS-RVND variando nº de clientes")
    ils_dir = os.path.dirname(__file__)
    ap.add_argument("--Q", type=int, default=100)
    ap.add_argument("--Y", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--datasets-dir", default=os.path.join(ils_dir, "datasets"))
    ap.add_argument("--output-dir",
                    default=os.path.join(ils_dir, "resultados", "cenarios_clientes"))
    ap.add_argument("--rapido", action="store_true")
    args = ap.parse_args()

    print("=== Cenários ILS-RVND — variando nº de clientes (dia 06-05, K=15) ===")
    resumos = []
    for idx, count in enumerate(COUNTS):
        dados = os.path.join(args.datasets_dir, f"dadosPO_misto_{count:03d}cli.xlsx")
        # offset disjunto por nível; pedidos/dia úteis ~2*count p/ cobrir os N locais
        print(f"\n--- gerando dataset: {count} clientes sintéticos ---")
        gerar_dataset(count, dados, seed=args.seed, offset=idx * 1000,
                      por_dia_util=max(40, 2 * count), por_fim_semana=12)

        n, n_locais = _contar_locais(dados, DATA, args.Q, args.Y)
        sc = {
            "nome": f"{count:03d}clientes",
            "titulo": f"{count} clientes sintéticos — 1 dia (06-05), K={K}",
            "K": K, "data_inicio": DATA, "data_fim": DATA,
            "runs": 3, "maxiter": 2, "maxiterils": 10,
        }
        r = run_scenario(sc, args.output_dir, dados, -19.9, -44.1,
                         args.Q, args.Y, args.seed, rapido=args.rapido)
        r["n_clientes"] = count
        r["locais"] = n_locais
        resumos.append(r)

    print("\n========== RESUMO — VARIANDO Nº DE CLIENTES ==========")
    print(f"{'clientes':>8s} {'nós':>5s} {'locais':>6s} {'melhor':>9s} "
          f"{'média':>9s} {'desvio':>7s} {'rotas':>6s} {'tempo/run':>9s}")
    for r in resumos:
        print(f"{r['n_clientes']:8d} {r['nos']:5d} {r['locais']:6d} "
              f"{r['melhor_custo']:9.1f} {r['media_custo']:9.1f} "
              f"{r['desvio_custo']:7.2f} {r['rotas']:6d} {r['tempo_medio_s']:8.1f}s")
    print(f"\nGráficos e CSVs por nível em: {args.output_dir}")


if __name__ == "__main__":
    main()
