"""Diferenciacao automatica matricial com CuPy.

Reutiliza EXATAMENTE a classe Value de autodiff_matricial.py.
O algoritmo de reverse-mode nao e reescrito: apenas o backend e alterado.

Exemplo:
    python autodiff_cupy.py
"""

import numpy as np
import cupy as cp

from autodiff_matricial import Value


def criar_grafo(X, W, b, backend=cp):
    """L = mean((X @ W + b)**2); retorna perda, X, W, b como Values."""
    x = Value(X, backend=backend)
    w = Value(W, backend=backend)
    vies = Value(b, backend=backend)
    loss = ((x @ w + vies) ** 2).mean()
    return loss, x, w, vies


def executar_gpu(X, W, b):
    """X, W e b ja devem ser arrays CuPy alocados na GPU."""
    loss, x, w, vies = criar_grafo(X, W, b, backend=cp)
    loss.backward()
    cp.cuda.get_current_stream().synchronize()
    return loss, x, w, vies


if __name__ == "__main__":
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("Nenhuma GPU CUDA disponivel")
    rng = np.random.default_rng(42)
    X = cp.asarray(rng.normal(size=(100, 32)), dtype=cp.float64)
    W = cp.asarray(rng.normal(size=(32, 16)), dtype=cp.float64)
    b = cp.zeros((1, 16), dtype=cp.float64)
    perda, x, w, vies = executar_gpu(X, W, b)
    print("Loss:", float(cp.asnumpy(perda.data)))
    print("dX:", x.grad.shape, "dW:", w.grad.shape, "db:", vies.grad.shape)
    print("dtype:", x.data.dtype)
