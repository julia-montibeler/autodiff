"""Benchmark de diferenca automatica matricial NumPy (CPU) vs CuPy (GPU).

Mesma classe Value, mesmas operacoes, mesmos dados e mesmo dtype float64.
Mede tempo total de parede (forward + backward) com sincronizacao da GPU.
As copias CPU<->GPU sao excluidas e devem ser documentadas no TCC.

Uso:
    python comparar_numpy_cupy.py
    python comparar_numpy_cupy.py --perfil grande --repeticoes 10

Arquivos em resultados_backprop_cpu_gpu/:
    medicoes_brutas_cpu_gpu.csv
    resumo_cpu_gpu.csv
    ambiente_cpu_gpu.json
"""

import argparse
import csv
import gc
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

try:
    import cupy as cp
except ImportError as exc:
    raise SystemExit(
        "CuPy nao instalado. Consulte https://docs.cupy.dev/en/latest/install.html "
        "para instalar a versao compativel com sua versao de CUDA."
    ) from exc

from autodiff_cupy import criar_grafo

SEED = 42
DTYPE = np.float64
CONFIG_PADRAO = [
    (64, 24, 12),
    (256, 128, 64),
    (1000, 256, 128),
    (2000, 512, 256),
]
CONFIG_GRANDE = [
    (1000, 100, 50),
    (5000, 200, 100),
    (10000, 500, 200),
]
ETAPAS = ("forward_ms", "backward_ms", "total_ms")


def sincronizar(backend):
    if backend is cp:
        cp.cuda.get_current_stream().synchronize()


def converter_para_numpy(valor, backend):
    return cp.asnumpy(valor) if backend is cp else np.asarray(valor)


def validar(X, W, b, Xgpu, Wgpu, bgpu):
    """Confere CPU e GPU contra derivadas analiticas em float64."""
    Z = X @ W + b
    delta = 2.0 * Z / Z.size
    esperado = {
        "loss": np.asarray(np.mean(Z ** 2)),
        "dX": delta @ W.T,
        "dW": X.T @ delta,
        "db": delta.sum(axis=0, keepdims=True),
    }
    for nome, backend, dados in (
        ("numpy_cpu", np, (X, W, b)),
        ("cupy_gpu", cp, (Xgpu, Wgpu, bgpu)),
    ):
        perda, x, w, vies = criar_grafo(*dados, backend=backend)
        for nome_array, valor in (("X", x), ("W", w), ("b", vies)):
            if valor.data.dtype != np.dtype("float64"):
                raise RuntimeError(
                    f"{nome}/{nome_array} esta em {valor.data.dtype}, mas o experimento "
                    "exige float64. Atualize autodiff_matricial.py."
                )
        perda.backward()
        sincronizar(backend)
        obtido = {
            "loss": converter_para_numpy(perda.data, backend),
            "dX": converter_para_numpy(x.grad, backend),
            "dW": converter_para_numpy(w.grad, backend),
            "db": converter_para_numpy(vies.grad, backend),
        }
        for campo in esperado:
            np.testing.assert_allclose(
                obtido[campo], esperado[campo], rtol=1e-9, atol=1e-10,
                err_msg=f"Diferenca em {nome}/{campo}",
            )
    return float(esperado["loss"])


def medir(backend, X, W, b):
    """Wall-clock de trabalho concluido; nao inclui alocacao/copia de entradas."""
    gc.collect()  # fora da regiao medida
    sincronizar(backend)
    gc_ligado = gc.isenabled()
    if gc_ligado:
        gc.disable()
    try:
        t0 = time.perf_counter_ns()
        perda, x, w, vies = criar_grafo(X, W, b, backend=backend)
        sincronizar(backend)  # GPU: espera o forward terminar
        t1 = time.perf_counter_ns()
        perda.backward()
        sincronizar(backend)  # GPU: espera o backward terminar
        t2 = time.perf_counter_ns()
    finally:
        if gc_ligado:
            gc.enable()
    return {
        "forward_ms": (t1 - t0) / 1e6,
        "backward_ms": (t2 - t1) / 1e6,
        "total_ms": (t2 - t0) / 1e6,
    }


def estatisticas(amostras):
    a = np.asarray(amostras, dtype=float)
    p25, p75 = np.percentile(a, (25, 75))
    return {
        "mediana": float(np.median(a)),
        "p25": float(p25),
        "p75": float(p75),
        "iqr": float(p75 - p25),
        "minimo": float(a.min()),
        "maximo": float(a.max()),
    }


def salvar_csv(caminho, linhas, campos):
    with caminho.open("w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(linhas)


def dados_ambiente(args, config):
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    gpu_nome = props.get("name", "desconhecida")
    if isinstance(gpu_nome, bytes):
        gpu_nome = gpu_nome.decode("utf-8", errors="replace")
    return {
        "seed": SEED,
        "dtype": "float64",
        "repeticoes": args.repeticoes,
        "aquecimentos": args.aquecimentos,
        "configuracoes": config,
        "metrica": "tempo de parede sincronizado, forward + backward",
        "transferencias": "H2D e D2H excluidas da regiao cronometrada",
        "tempo": "time.perf_counter_ns, stream CUDA sincronizado apos forward e backward",
        "ordem": "alternada (CPU/GPU e GPU/CPU)",
        "gc": "gc.collect antes, GC desabilitado no intervalo cronometrado",
        "python": sys.version,
        "numpy": np.__version__,
        "cupy": cp.__version__,
        "plataforma": platform.platform(),
        "cpu": platform.processor() or platform.uname().processor,
        "cpu_logicos": os.cpu_count(),
        "gpu": gpu_nome,
        "cuda_runtime_version": cp.cuda.runtime.runtimeGetVersion(),
        "threads": {k: os.environ.get(k) for k in (
            "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perfil", choices=["padrao", "grande"], default="padrao")
    parser.add_argument("--repeticoes", type=int, default=30)
    parser.add_argument("--aquecimentos", type=int, default=5)
    parser.add_argument("--saida", type=Path, default=Path("resultados_backprop_cpu_gpu"))
    args = parser.parse_args()
    if args.repeticoes < 1 or args.aquecimentos < 0:
        parser.error("repeticoes >= 1 e aquecimentos >= 0 sao obrigatorios")

    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("Nao foi encontrada uma GPU CUDA")
    except cp.cuda.runtime.CUDARuntimeError as exc:
        raise SystemExit(f"Nao foi possivel acessar a GPU CUDA: {exc}") from exc

    config = CONFIG_PADRAO if args.perfil == "padrao" else CONFIG_GRANDE
    args.saida.mkdir(exist_ok=True, parents=True)
    rng = np.random.default_rng(SEED)
    brutos, resumo = [], []

    for n, entradas, saidas in config:
        X = rng.normal(size=(n, entradas)).astype(DTYPE)
        W = (rng.normal(size=(entradas, saidas)) / np.sqrt(entradas)).astype(DTYPE)
        b = rng.normal(size=(1, saidas)).astype(DTYPE)
        # Copia para a GPU UMA VEZ, sempre fora das regioes de medicao.
        Xgpu, Wgpu, bgpu = cp.asarray(X), cp.asarray(W), cp.asarray(b)
        sincronizar(cp)

        loss = validar(X, W, b, Xgpu, Wgpu, bgpu)
        print(f"\nX=({n}, {entradas}), W=({entradas}, {saidas}) | loss={loss:.6f}")

        series = {"numpy_cpu": [], "cupy_gpu": []}
        for rodada in range(args.aquecimentos + args.repeticoes):
            ordem = ("numpy_cpu", "cupy_gpu") if rodada % 2 == 0 else ("cupy_gpu", "numpy_cpu")
            for posicao, versao in enumerate(ordem, start=1):
                backend = np if versao == "numpy_cpu" else cp
                entradas_dados = (X, W, b) if backend is np else (Xgpu, Wgpu, bgpu)
                valores = medir(backend, *entradas_dados)
                if rodada >= args.aquecimentos:
                    serie = {"amostras": n, "entradas": entradas, "saidas": saidas,
                             "versao": versao, "repeticao": rodada - args.aquecimentos + 1,
                             "posicao_na_rodada": posicao, **valores}
                    brutos.append(serie)
                    series[versao].append(valores)

        linha = {"amostras": n, "entradas": entradas, "saidas": saidas,
                 "loss": loss, "dtype": "float64"}
        for versao in ("numpy_cpu", "cupy_gpu"):
            for etapa in ETAPAS:
                stats = estatisticas([m[etapa] for m in series[versao]])
                for nome, valor in stats.items():
                    linha[f"{etapa.removesuffix('_ms')}_{versao}_{nome}_ms"] = valor
        tcpu = linha["total_numpy_cpu_mediana_ms"]
        tgpu = linha["total_cupy_gpu_mediana_ms"]
        linha["aceleracao_gpu_sobre_cpu_x"] = tcpu / tgpu
        resumo.append(linha)
        print(f"  CPU (NumPy): {tcpu:.3f} ms | GPU (CuPy): {tgpu:.3f} ms")
        print(f"  Aceleracao GPU/CPU: {tcpu/tgpu:.2f}x")
        print("  (>1 significa GPU mais rapida; <1 significa CPU mais rapida)")

        # Libera as referencias das entradas desse tamanho antes da proxima configuracao.
        del Xgpu, Wgpu, bgpu

    salvar_csv(args.saida / "medicoes_brutas_cpu_gpu.csv", brutos, list(brutos[0]))
    salvar_csv(args.saida / "resumo_cpu_gpu.csv", resumo, list(resumo[0]))
    (args.saida / "ambiente_cpu_gpu.json").write_text(
        json.dumps(dados_ambiente(args, config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nArquivos salvos em: {args.saida.resolve()}")


if __name__ == "__main__":
    main()
