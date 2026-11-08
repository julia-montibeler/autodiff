
import math
from pathlib import Path

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self._backward = lambda: None
        self.grad = 0.0
        self.label = label
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data**other, (self, ), f'**{other}')
        def _backward():
            self.grad += other * self.data**(other - 1) * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(t, (self, ), 'tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad
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
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges


def draw_dot(root):
    from graphviz import Digraph

    dot = Digraph(
        format='svg',
        graph_attr={
            'rankdir': 'TB',
            'nodesep': '0.28',
            'ranksep': '0.35',
            'pad': '0.15'
        }
    )

    dot.attr(
        'node',
        fontname='Helvetica',
        fontsize='10'
    )

    nodes, edges = trace(root)

    for n in sorted(nodes, key=lambda v: (v.label, id(v))):
        uid = str(id(n))

        dot.node(
            name=uid,
            label="{ %s | data %.4f | grad %.4f }" %
                  (n.label, n.data, n.grad),
            shape='record'
        )

        if n._op:
            dot.node(
                name=uid + n._op,
                label=n._op
            )
            dot.edge(uid + n._op, uid)

    for n1, n2 in sorted(
        edges,
        key=lambda edge: (
            edge[1].label,
            edge[0].label,
            id(edge[1]),
            id(edge[0])
        )
    ):
        dot.edge(
            str(id(n1)),
            str(id(n2)) + n2._op
        )

    return dot


def salvar_grafo(root, nome, pasta='lib'):
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)

    dot = draw_dot(root)

    for formato in ('png', 'pdf'):
        dot.render(
            filename=nome,
            directory=str(pasta),
            format=formato,
            view=False,
            cleanup=True
        )


def main():
    x = Value(2.0, label='x')
    w = Value(-3.0, label='w')
    b = Value(5.0, label='b')

    xw = x * w
    xw.label = 'x*w'

    n = xw + b
    n.label = 'n = x*w + b'

    loss = n ** 2
    loss.label = 'L = n^2'

    salvar_grafo(
        loss,
        'grafo_passagem_direta'
    )

    loss.backward()

    salvar_grafo(
        loss,
        'grafo_exemplo_simples'
    )

    print("Valor de n:", n.data)
    print("Valor de L:", loss.data)
    print("Gradiente de x:", x.grad)
    print("Gradiente de w:", w.grad)
    print("Gradiente de b:", b.grad)


if __name__ == "__main__":
    main()
