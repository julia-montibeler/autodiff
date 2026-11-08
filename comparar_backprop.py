"""Comparacao na CPU entre autodiff escalar e autodiff matricial (NumPy).

Reutiliza as implementacoes existentes, sem redefinir o algoritmo:
    backpropagation.py      -> class Value
    autodiff_matricial.py   -> class Value

Experimento: L = mean((X @ W + b)**2)
Mede construcao/forward, backward e tempo total com perf_counter_ns.
Valida a perda e os gradientes contra derivadas analiticas.
Repeticoes, aquecimentos, IQR e metadados de ambiente para reproducibilidade.

Execute:
    python comparar_backprop.py
"""

import csv
import gc
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from backpropagation import Value as ValueEscalar
from autodiff_matricial import Value as ValueMatricial

SEED = 42
DTYPE = np.float64  # Mesma precisao para as duas implementacoes
REPETICOES = 30
AQUECIMENTOS = 5
CONFIGURACOES = [
    (8, 8, 4),
    (16, 12, 6),
    (32, 16, 8),
    (64, 24, 12),
]
PASTA_RESULTADOS = Path("resultados_backprop_cpu_float64")
CAMPOS_TEMPO = ("forward_ms", "backward_ms", "total_ms")


def criar_grafo_escalar(X, W, b):
    n, entradas = X.shape
    saidas = W.shape[1]
    x = [[ValueEscalar(float(X[i, k])) for k in range(entradas)]
         for i in range(n)]
    w = [[ValueEscalar(float(W[k, j])) for j in range(saidas)]
         for k in range(entradas)]
    vies = [ValueEscalar(float(b[0, j])) for j in range(saidas)]

    soma_quadrados = None
    for i in range(n):
        for j in range(saidas):
            z = x[i][0] * w[0][j]
            for k in range(1, entradas):
                z = z + x[i][k] * w[k][j]
            z = z + vies[j]
            quadrado = z ** 2
            soma_quadrados = (quadrado if soma_quadrados is None
                              else soma_quadrados + quadrado)

    perda = soma_quadrados * (1.0 / (n * saidas))
    return perda, x, w, vies


def criar_grafo_matricial(X, W, b):
    x = ValueMatricial(X)
    w = ValueMatricial(W)
    vies = ValueMatricial(b)
    perda = ((x @ w + vies) ** 2).mean()
    return perda, x, w, vies


def extrair_resultados(grafo, versao):
    perda, x, w, b = grafo
    if versao == "escalar":
        return (
            float(perda.data),
            np.array([[v.grad for v in linha] for linha in x], dtype=np.float64),
            np.array([[v.grad for v in linha] for linha in w], dtype=np.float64),
            np.array([[v.grad for v in b]], dtype=np.float64),
        )
    return (
        float(np.asarray(perda.data)),
        np.asarray(x.grad, dtype=np.float64),
        np.asarray(w.grad, dtype=np.float64),
        np.asarray(b.grad, dtype=np.float64),
    )


def validar(X, W, b):
    """Valida AMBAS as implementacoes contra derivadas independentes."""
    # Referencia calculada em float64 para reduzir erro de arredondamento.
    X64 = X.astype(np.float64)
    W64 = W.astype(np.float64)
    b64 = b.astype(np.float64)
    Z = X64 @ W64 + b64
    delta = 2.0 * Z / Z.size
    referencia = (
        float(np.mean(Z ** 2)),
        delta @ W64.T,
        X64.T @ delta,
        delta.sum(axis=0, keepdims=True),
    )
    for nome, construir in (
        ("escalar", criar_grafo_escalar),
        ("matricial", criar_grafo_matricial),
    ):
        grafo = construir(X, W, b)
        grafo[0].backward()
        resultado = extrair_resultados(grafo, nome)
        for campo, obtido, esperado in zip(("loss", "dX", "dW", "db"), resultado, referencia):
            try:
                np.testing.assert_allclose(
                    obtido, esperado, rtol=1e-3, atol=1e-4,
                    err_msg=f"Falha no {campo} da implementacao {nome}",
                )
            except AssertionError as exc:
                raise AssertionError(f"{nome} / {campo}: {exc}") from exc
    return referencia[0]


def contar_nos(raiz):
    """Numero de objetos/nos alcançaveis pelo grafo (fora da medicao)."""
    visitados = set()
    pilha = [raiz]
    while pilha:
        no = pilha.pop()
        if id(no) in visitados:
            continue
        visitados.add(id(no))
        pilha.extend(getattr(no, "_prev", ()))
    return len(visitados)


def medir(versao, X, W, b):
    # Coleta entre execucoes, NUNCA dentro da regiao cronometrada.
    gc.collect()
    estava_ativo = gc.isenabled()
    if estava_ativo:
        gc.disable()
    try:
        inicio = time.perf_counter_ns()
        if versao == "escalar":
            perda, *_ = criar_grafo_escalar(X, W, b)
        else:
            perda, *_ = criar_grafo_matricial(X, W, b)
        fim_forward = time.perf_counter_ns()

        perda.backward()
        fim_backward = time.perf_counter_ns()
    finally:
        if estava_ativo:
            gc.enable()

    return {
        "forward_ms": (fim_forward - inicio) / 1_000_000,
        "backward_ms": (fim_backward - fim_forward) / 1_000_000,
        "total_ms": (fim_backward - inicio) / 1_000_000,
    }


def resumo_medidas(valores):
    valores = np.asarray(valores, dtype=float)
    p25, p75 = np.percentile(valores, [25, 75])
    return {
        "mediana": float(np.median(valores)),
        "p25": float(p25),
        "p75": float(p75),
        "iqr": float(p75 - p25),
        "minimo": float(np.min(valores)),
        "maximo": float(np.max(valores)),
    }


def salvar_csv(caminho, dados, colunas):
    with caminho.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=colunas)
        writer.writeheader()
        writer.writerows(dados)


def metadados():
    return {
        "seed": SEED,
        "dtype": np.dtype(DTYPE).name,
        "repeticoes": REPETICOES,
        "aquecimentos": AQUECIMENTOS,
        "configuracoes": CONFIGURACOES,
        "relogio": "time.perf_counter_ns (monotonico, tempo decorrido)",
        "metrica": "construcao do grafo + backward, sem geracao de dados e sem CSV",
        "gc": "gc.collect antes, GC desabilitado durante a medicao",
        "ordem": "alternada (escalar/matricial, matricial/escalar)",
        "python": sys.version,
        "numpy": np.__version__,
        "plataforma": platform.platform(),
        "processador": platform.processor() or platform.uname().processor,
        "cpu_logicos": os.cpu_count(),
        "variaveis_threads": {
            chave: os.environ.get(chave)
            for chave in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
        },
        "precisao": "float64 (entradas e Value matricial; escalar recebe float Python, precisao dupla)",
    }


def main():
    PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    brutos = []
    resumo = []

    for n, entradas, saidas in CONFIGURACOES:
        # Mesmos operandos para as duas versoes; criacao FORA da medicao.
        X = rng.normal(size=(n, entradas)).astype(DTYPE)
        W = (rng.normal(size=(entradas, saidas)) / np.sqrt(entradas)).astype(DTYPE)
        b = rng.normal(size=(1, saidas)).astype(DTYPE)

        loss = validar(X, W, b)
        nos_escalar = contar_nos(criar_grafo_escalar(X, W, b)[0])
        nos_matricial = contar_nos(criar_grafo_matricial(X, W, b)[0])
        gc.collect()
        print(f"\nX=({n}, {entradas}), W=({entradas}, {saidas}) | loss={loss:.6f}")
        print(f"  Nos: escalar={nos_escalar}, matricial={nos_matricial}")

        tempos = {"escalar": [], "matricial": []}
        for rodada in range(AQUECIMENTOS + REPETICOES):
            ordem = ("escalar", "matricial") if rodada % 2 == 0 else ("matricial", "escalar")
            for posicao, versao in enumerate(ordem, start=1):
                medida = medir(versao, X, W, b)
                if rodada >= AQUECIMENTOS:
                    tempos[versao].append(medida)
                    brutos.append({
                        "amostras": n,
                        "entradas": entradas,
                        "saidas": saidas,
                        "versao": versao,
                        "repeticao": rodada - AQUECIMENTOS + 1,
                        "posicao_na_rodada": posicao,
                        **medida,
                    })

        stats = {
            versao: {
                campo: resumo_medidas([medida[campo] for medida in lista])
                for campo in CAMPOS_TEMPO
            }
            for versao, lista in tempos.items()
        }
        esc, mat = stats["escalar"], stats["matricial"]
        ganho = esc["total_ms"]["mediana"] / mat["total_ms"]["mediana"]
        linha = {
            "amostras": n,
            "entradas": entradas,
            "saidas": saidas,
            "loss": loss,
            "nos_escalar": nos_escalar,
            "nos_matricial": nos_matricial,
            "aceleracao_total_x": ganho,
        }
        for versao, sufixo in (("escalar", "escalar"), ("matricial", "matricial")):
            for campo in CAMPOS_TEMPO:
                nome = campo.removesuffix("_ms")
                for estatistica, valor in stats[versao][campo].items():
                    linha[f"{nome}_{sufixo}_{estatistica}_ms"] = valor
        resumo.append(linha)

        print(f"  Escalar:   {esc['total_ms']['mediana']:.3f} ms "
              f"(IQR {esc['total_ms']['iqr']:.3f} ms)")
        print(f"  Matricial: {mat['total_ms']['mediana']:.3f} ms "
              f"(IQR {mat['total_ms']['iqr']:.3f} ms)")
        print(f"  Aceleracao (razao das medianas): {ganho:.2f}x")

    salvar_csv(
        PASTA_RESULTADOS / "medicoes_brutas.csv", brutos,
        ["amostras", "entradas", "saidas", "versao", "repeticao",
         "posicao_na_rodada", *CAMPOS_TEMPO],
    )
    salvar_csv(
        PASTA_RESULTADOS / "comparacao_escalar_matricial.csv", resumo,
        list(resumo[0].keys()),
    )
    with (PASTA_RESULTADOS / "ambiente.json").open("w", encoding="utf-8") as f:
        json.dump(metadados(), f, ensure_ascii=False, indent=2)
    print(f"\nCSVs e ambiente salvos em: {PASTA_RESULTADOS.resolve()}")


if __name__ == "__main__":
    main()
