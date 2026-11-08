"""Classe Value matricial usando float64 para comparação controlada com a versão escalar."""
import numpy as np


class Value:
    def __init__(self, data, _children=(), _op="", backend=np):
        self.xp = backend
        self.data = self.xp.asarray(data, dtype=self.xp.float64)
        self.grad = self.xp.zeros_like(self.data)

        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    @staticmethod
    def _unbroadcast(grad, shape):
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)

        for axis, size in enumerate(shape):
            if size == 1:
                grad = grad.sum(axis=axis, keepdims=True)
        return grad

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other, backend=self.xp)
        out = Value(self.data + other.data, (self, other), "+", backend=self.xp)

        def _backward():
            self.grad += self._unbroadcast(out.grad, self.data.shape)
            other.grad += self._unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other, backend=self.xp)
        out = Value(self.data * other.data, (self, other), "*", backend=self.xp)

        def _backward():
            self.grad += self._unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += self._unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __matmul__(self, other):
        other = other if isinstance(other, Value) else Value(other, backend=self.xp)
        out = Value(self.data @ other.data, (self, other), "@", backend=self.xp)

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def __pow__(self, exponent):
        out = Value(self.data ** exponent, (self,), f"**{exponent}", backend=self.xp)

        def _backward():
            self.grad += exponent * self.data ** (exponent - 1) * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other if isinstance(other, Value) else -other)

    def __rsub__(self, other):
        return other + (-self)

    def mean(self):
        out = Value(self.xp.mean(self.data), (self,), "mean", backend=self.xp)

        def _backward():
            self.grad += self.xp.ones_like(self.data) * (out.grad / self.data.size)

        out._backward = _backward
        return out

    def tanh(self):
        t = self.xp.tanh(self.data)
        out = Value(t, (self,), "tanh", backend=self.xp)

        def _backward():
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        for node in topo:
            node.grad = node.xp.zeros_like(node.data)

        self.grad = self.xp.ones_like(self.data)
        for node in reversed(topo):
            node._backward()


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    X = Value(rng.random((1000, 100), dtype=np.float64))
    W = Value(rng.random((100, 50), dtype=np.float64))
    b = Value(np.zeros((1, 50), dtype=np.float64))
    loss = ((X @ W + b) ** 2).mean()
    loss.backward()
    print("Loss:", loss.data)
    print("Gradiente X:", X.grad.shape)
    print("Gradiente W:", W.grad.shape)
    print("Gradiente b:", b.grad.shape)
