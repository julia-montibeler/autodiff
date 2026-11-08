"""MLP 2 -> 3 -> 1 com diferenciação automática e geração dos resultados do TCC."""

import csv
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Backpropagation.backpropagation import Value


PASTA_RESULTADOS = Path(__file__).resolve().parent / "lib"


class Neuronio:
    """Um neurônio; ativação tanh (oculto) ou identidade (saída)."""

    def __init__(self, n_entradas, rng, ativacao="tanh"):
        if ativacao not in ("tanh", "linear"):
            raise ValueError("Ativação deve ser 'tanh' ou 'linear'.")
        self.pesos = [Value(float(rng.uniform(-0.6, 0.6)), label=f"w{i}")
                      for i in range(n_entradas)]
        self.vies = Value(0.0, label="b")
        self.ativacao = ativacao

    def __call__(self, entradas):
        if len(entradas) != len(self.pesos):
            raise ValueError("Número de entradas incompatível com o neurônio")
        z = self.vies
        for w, x in zip(self.pesos, entradas):
            z = z + w * x
        return z.tanh() if self.ativacao == "tanh" else z

    def parametros(self):
        return self.pesos + [self.vies]


class Camada:
    def __init__(self, n_entradas, n_neuronios, rng, ativacao):
        self.neuronios = [Neuronio(n_entradas, rng, ativacao)
                          for _ in range(n_neuronios)]

    def __call__(self, entradas):
        return [n(entradas) for n in self.neuronios]

    def parametros(self):
        return [p for n in self.neuronios for p in n.parametros()]


class MLP:
    """Rede de 2 entradas -> 3 neurônios tanh -> 1 neurônio linear."""

    def __init__(self, arquitetura=(2, 3, 1), seed=42):
        if len(arquitetura) < 2 or any(n < 1 for n in arquitetura):
            raise ValueError("Arquitetura deve conter inteiros positivos")
        rng = np.random.default_rng(seed)
        self.camadas = []
        for i, (nin, nout) in enumerate(zip(arquitetura[:-1], arquitetura[1:])):
            ativacao = "linear" if i == len(arquitetura) - 2 else "tanh"
            self.camadas.append(Camada(nin, nout, rng, ativacao))

    def __call__(self, entradas):
        valores = [v if isinstance(v, Value) else Value(float(v))
                   for v in entradas]
        for camada in self.camadas:
            valores = camada(valores)
        return valores[0] if len(valores) == 1 else valores

    def parametros(self):
        return [p for camada in self.camadas for p in camada.parametros()]

    def prever(self, X):
        X = np.asarray(X, dtype=float)
        X = np.atleast_2d(X)
        return np.array([self(linha).data for linha in X])

    def passo(self, x, alvo, taxa_aprendizagem):
        saida = self(x)
        perda = (saida - float(alvo)) ** 2
        parametros = self.parametros()
        for p in parametros:
            p.grad = 0.0
        perda.backward()
        for p in parametros:
            p.data -= taxa_aprendizagem * p.grad
        return float(perda.data)

    def treinar(self, X, y, epocas=1200, taxa_aprendizagem=0.02, seed=7):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(self.camadas[0].neuronios[0].pesos):
            raise ValueError("Formato incorreto de X")
        if y.shape != (len(X),) or len(X) == 0 or epocas <= 0 or taxa_aprendizagem <= 0:
            raise ValueError("Dados ou hiperparâmetros inválidos")
        rng = np.random.default_rng(seed)
        historico = []
        for epoca in range(1, epocas + 1):
            for idx in rng.permutation(len(X)):
                self.passo(X[idx], y[idx], taxa_aprendizagem)
            mse = float(np.mean((self.prever(X) - y) ** 2))
            historico.append(mse)
            if epoca == 1 or epoca % 200 == 0 or epoca == epocas:
                print(f"Época {epoca:4d} | MSE treinamento: {mse:.6f}", flush=True)
        return historico


def testar_gradiente():
    rede = MLP(seed=11)
    x = [0.35, -0.4]
    alvo = 0.5
    parametros = rede.parametros()
    for p in parametros:
        p.grad = 0.0
    perda = (rede(x) - alvo) ** 2
    perda.backward()
    automaticos = [p.grad for p in parametros]
    h = 1e-6
    linhas = []
    nomes = ["w11", "w12", "b1", "w21", "w22", "b2", "w31", "w32", "b3", "v1", "v2", "v3", "c"]
    for i, p in enumerate(parametros):
        original = p.data
        p.data = original + h
        lp = (rede(x).data - alvo) ** 2
        p.data = original - h
        lm = (rede(x).data - alvo) ** 2
        p.data = original
        numerico = (lp - lm) / (2 * h)
        if not math.isclose(automaticos[i], numerico, rel_tol=1e-4, abs_tol=1e-6):
            raise AssertionError(f"Parâmetro {i}: autodiff={automaticos[i]}, numérico={numerico}")
        linhas.append((nomes[i], automaticos[i], numerico, abs(automaticos[i] - numerico)))
    print(f"Teste de gradientes: OK ({len(parametros)}/{len(parametros)})", flush=True)
    print(f"Maior diferença absoluta: {max(row[3] for row in linhas):.3e}", flush=True)
    return linhas


def gerar_arquitetura():
    fig, ax = plt.subplots(figsize=(8, 4.5))
    camadas = [
        [(0, 0.6), (0, -0.6)],
        [(1.6, 1), (1.6, 0), (1.6, -1)],
        [(3.2, 0)]
    ]
    rotulos = [
        [r'$x_1$', r'$x_2$'],
        [r'$h_1$', r'$h_2$', r'$h_3$'],
        [r'$\hat{y}$']
    ]
    for origem, destino in zip(camadas[:-1], camadas[1:]):
        for x0, y0 in origem:
            for x1, y1 in destino:
                ax.plot([x0, x1], [y0, y1], linewidth=1, zorder=1)
    for posicoes, nomes in zip(camadas, rotulos):
        for (x, y), nome in zip(posicoes, nomes):
            ax.add_patch(plt.Circle((x, y), 0.21, facecolor='white',
                                    edgecolor='black', linewidth=1.8, zorder=3))
            ax.text(x, y, nome, ha='center', va='center', fontsize=13, zorder=4)
    for x, titulo in [(0, 'Entrada'), (1.6, 'Camada oculta (tanh)'), (3.2, 'Saída linear')]:
        ax.text(x, 1.45, titulo, ha='center', va='center', fontsize=11)
    ax.set_xlim(-0.6, 3.8)
    ax.set_ylim(-1.45, 1.65)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(PASTA_RESULTADOS / 'mlp_arquitetura_231.png', dpi=220)
    plt.close(fig)


def gerar_grafico_erro(historico):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(np.arange(1, len(historico) + 1), historico)
    ax.set(title='MLP 2-3-1 com autodiff — erro de treinamento',
           xlabel='Época', ylabel='MSE')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PASTA_RESULTADOS / 'mlp_curva_erro.png', dpi=220)
    plt.close(fig)


def gerar_grafico_previsoes(esperados, previstos):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(esperados, previstos, alpha=0.8)
    limite_min = float(min(np.min(esperados), np.min(previstos)))
    limite_max = float(max(np.max(esperados), np.max(previstos)))
    ax.plot([limite_min, limite_max], [limite_min, limite_max], '--', linewidth=1)
    ax.set(xlabel='Valor esperado', ylabel='Valor previsto',
           title='MLP 2-3-1 — previsões no conjunto de teste')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PASTA_RESULTADOS / 'mlp_previsoes_teste.png', dpi=220)
    plt.close(fig)


def salvar_tabelas(historico, X_teste, y_teste, pred_teste, gradientes):
    with (PASTA_RESULTADOS / 'mlp_mse_epocas.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['epoca', 'mse_treinamento'])
        writer.writerows((i, mse) for i, mse in enumerate(historico, start=1))

    with (PASTA_RESULTADOS / 'mlp_previsoes_teste.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['x1', 'x2', 'esperado', 'previsto'])
        writer.writerows((x[0], x[1], alvo, pred)
                         for x, alvo, pred in zip(X_teste, y_teste, pred_teste))

    with (PASTA_RESULTADOS / 'mlp_verificacao_gradientes.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['parametro', 'gradiente_automatico', 'gradiente_central', 'diferenca_absoluta'])
        writer.writerows(gradientes)


def main():
    print('INICIANDO EXPERIMENTO MLP', flush=True)
    PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    gradientes = testar_gradiente()
    gerar_arquitetura()

    eixo = np.linspace(-1, 1, 11)
    X = np.array([(a, b) for a in eixo for b in eixo], dtype=float)
    y = X[:, 0] ** 2 + X[:, 1] ** 2
    indices = np.random.default_rng(123).permutation(len(X))
    n_treino = int(0.8 * len(X))
    treino, teste = indices[:n_treino], indices[n_treino:]

    mlp = MLP(arquitetura=(2, 3, 1), seed=42)
    print(f'Rede 2-3-1 com {len(mlp.parametros())} parâmetros', flush=True)
    historico = mlp.treinar(X[treino], y[treino])

    pred_teste = mlp.prever(X[teste])
    mse_teste = float(np.mean((pred_teste - y[teste]) ** 2))
    print(f'MSE no teste: {mse_teste:.6f}', flush=True)
    print('Exemplos de previsão em teste:')
    for i in range(min(5, len(teste))):
        print(f'  x={X[teste][i]} | esperado={y[teste][i]:.4f} | previsto={pred_teste[i]:.4f}')

    gerar_grafico_erro(historico)
    gerar_grafico_previsoes(y[teste], pred_teste)
    salvar_tabelas(historico, X[teste], y[teste], pred_teste, gradientes)
    print(f'Figuras e tabelas salvas em: {PASTA_RESULTADOS.resolve()}', flush=True)


if __name__ == '__main__':
    main()
