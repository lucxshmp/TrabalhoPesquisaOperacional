import os
import time
import signal
import traceback
import pandas as pd
import numpy as np
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from main import run_model
import csv
import matplotlib.pyplot as plt

# ===============================
# CONFIG
# ===============================

DATASET_DIR = r"C:\Users\LUCAS\Documents\Faculdade\Pesquisa Operacional\Trabalho\TrabalhoPesquisaOperacional\BCP\datasets\dados sintéticos"

CENARIOS = {
    "20": "dataset_20_clientes.xlsx",
    "50": "dataset_50_clientes.xlsx",
}

Q = 500
Y = 48
K = 10

OUTPUT_FILE = "resultados.csv"
TIMEOUT_SEGUNDOS = 300

# ===============================
# CRIA CSV SE NÃO EXISTIR
# ===============================

if not os.path.exists(OUTPUT_FILE):
    pd.DataFrame(columns=[
        "timestamp", "clientes", "arquivo", "status",
        "tempo", "colunas", "iteracoes", "rotas",
        "custo", "gap", "erro", "log"
    ]).to_csv(OUTPUT_FILE, index=False, quoting=csv.QUOTE_ALL)

# ===============================
# TIMEOUT
# ===============================

class TimeoutError(Exception):
    pass

def _handler(signum, frame):
    raise TimeoutError("Timeout atingido")

USE_SIGNAL = hasattr(signal, 'SIGALRM')

# ===============================
# EXECUÇÃO DOS CENÁRIOS
# ===============================

for nome, arquivo in CENARIOS.items():

    path = os.path.join(DATASET_DIR, arquivo)
    print(f"\n🚀 Rodando cenário {nome} clientes")

    inicio = time.time()

    registro = {
        "timestamp": datetime.now().isoformat(),
        "clientes": int(nome),
        "arquivo": arquivo,
        "status": "OK",
        "tempo": None,
        "colunas": None,
        "iteracoes": None,
        "rotas": None,
        "custo": None,
        "gap": None,
        "erro": None,
        "log": None
    }

    stdout_buf = StringIO()
    stderr_buf = StringIO()

    try:
        if USE_SIGNAL:
            signal.signal(signal.SIGALRM, _handler)
            signal.alarm(TIMEOUT_SEGUNDOS)

        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            res = run_model(path, path, Q, Y, K)

        if USE_SIGNAL:
            signal.alarm(0)

        tempo = round(time.time() - inicio, 2)

        registro.update({
            "tempo": tempo,
            "colunas": res.get("colunas"),
            "iteracoes": res.get("iteracoes"),
            "rotas": res.get("rotas"),
            "custo": res.get("custo"),
            "gap": res.get("gap"),
            "log": stdout_buf.getvalue()[:500]
        })

        print(f"✅ OK | tempo={tempo}s")

    except TimeoutError:
        registro.update({
            "status": "TIMEOUT",
            "erro": f"Excedeu {TIMEOUT_SEGUNDOS}s",
            "tempo": round(time.time() - inicio, 2),
            "log": stdout_buf.getvalue()[:500]
        })
        print(f"⏱️ TIMEOUT")

    except Exception as e:
        registro.update({
            "status": "ERRO",
            "erro": str(e),
            "tempo": round(time.time() - inicio, 2),
            "log": stdout_buf.getvalue()[:300] + "\n---\n" + traceback.format_exc()[-300:]
        })
        print(f"❌ ERRO: {e}")

    finally:
        pd.DataFrame([registro]).to_csv(
            OUTPUT_FILE,
            mode='a',
            header=False,
            index=False,
            quoting=csv.QUOTE_ALL
        )
        print(f"💾 Salvo em {OUTPUT_FILE}")

# ===============================
# ANÁLISE FINAL
# ===============================

df = pd.read_csv(OUTPUT_FILE)

# filtra execuções válidas
df = df[df["status"] == "OK"]
df = df.sort_values("clientes")

print("\n📊 Resultados:")
print(df.to_string(index=False))

# ===============================
# FUNÇÃO DE PLOT
# ===============================

def plot(x, y, titulo, ylabel, nome_arquivo, log=False):
    plt.figure()
    plt.plot(x, y, marker='o')
    if log:
        plt.yscale("log")
    plt.title(titulo)
    plt.xlabel("Clientes")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(nome_arquivo, dpi=300)
    plt.show()

# ===============================
# MÉTRICAS DERIVADAS
# ===============================

df["tempo_iter"] = df["tempo"] / df["iteracoes"].replace(0, np.nan)
df["colunas_iter"] = df["colunas"] / df["iteracoes"].replace(0, np.nan)

# ===============================
# GRÁFICOS
# ===============================

plot(df["clientes"], df["tempo"],
     "Tempo vs Clientes", "Tempo (s)",
     "tempo_vs_clientes.png", log=True)

plot(df["clientes"], df["colunas"],
     "Colunas vs Clientes", "Colunas",
     "colunas_vs_clientes.png", log=True)

plot(df["clientes"], df["iteracoes"],
     "Iterações vs Clientes", "Iterações",
     "iteracoes_vs_clientes.png", log=True)

plot(df["clientes"], df["custo"],
     "Custo vs Clientes", "Custo",
     "custo_vs_clientes.png")

plot(df["clientes"], df["gap"],
     "Gap vs Clientes", "Gap",
     "gap_vs_clientes.png")

plot(df["clientes"], df["tempo_iter"],
     "Tempo por Iteração", "Tempo/Iter",
     "tempo_por_iteracao.png", log=True)

plot(df["clientes"], df["colunas_iter"],
     "Colunas por Iteração", "Colunas/Iter",
     "colunas_por_iteracao.png")