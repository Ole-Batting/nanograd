import math
import random

class Var:
    def __init__(self, val, parents=None):
        self.val = val
        self.parents = parents if parents else list()
        self.grad = 0.0

    def backprop(self, bp):
        self.grad += bp
        for parent, grad in self.parents:
            parent.backprop(grad * bp)

    def __add__(self: 'Var', other: 'Var'):
        return Var(self.val + other.val, [(self, 1.0), (other, 1.0)])

    def __mul__(self, other):
        return Var(self.val * other.val, [(self, other.val), (other, self.val)])

    def __neg__(self):
        return Var(-1.0) * self

    def __sub__(self, other):
        return self + (-other)

    def __pow__(self, power):
        return Var(self.val ** power, [(self, power * self.val ** (power - 1))])

    def __truediv__(self, other):
        return self * other ** -1

    def step(self):
        self.val -= self.grad

    def __repr__(self):
        return f"Var({self.val:.4f}, grad={self.grad:.4e})"

    @staticmethod
    def random(scale):
        return Var(2 * scale * (random.random() - 0.5))


def relu(a):
    return a if a.val > 0.0 else Var(0.0)


def demo_var_taylor(n_samples, n_coefs, n_epochs, lr, print_every):
    x_samples = [Var.random(2.0) for _ in range(n_samples)]
    y_samples = [Var(math.cos(x.val)) for x in x_samples]

    c_params = [Var.random(1.0) for _ in range(n_coefs)]

    def taylor(x):
        out = Var(0.0)
        for i, c in enumerate(c_params):
            out += c * x ** i
        return out

    y_predict = [Var(0.0) for _ in range(n_samples)]

    for epoch in range(n_epochs):
        loss = Var(0.0)
        for i, (x, y) in enumerate(zip(x_samples, y_samples)):
            y_hat = taylor(x)
            y_predict[i] = y_hat
            loss += (y - y_hat) ** 2
        loss /= Var(n_samples)

        if loss.val > 10000:
            print("loss exploded! maybe lower learning rate")
            break

        loss.backprop(lr)
        if i % print_every == 0:
            print(f"{epoch=:4d}: {loss.val=:.4e}")

        for c in c_params:
            c.step()
            c.grad = 0.0

    print(c_params)

    return x_samples, y_samples, y_predict


class Matrix:
    def __init__(self, values, n_rows, n_cols):
        self.values = values
        self.n_rows = n_rows
        self.n_cols = n_cols

    @staticmethod
    def zeros(n_rows, n_cols):
        return Matrix([Var(0.0) for _ in range(n_rows * n_cols)], n_rows, n_cols)

    @staticmethod
    def random(n_rows, n_cols, scale):
        return Matrix([Var.random(scale) for _ in range(n_rows * n_cols)], n_rows, n_cols)

    def __getitem__(self, ij):
        ith_row, jth_col = ij
        return self.values[ith_row * self.n_cols + jth_col]

    def __setitem__(self, ij, val):
        ith_row, jth_col = ij
        self.values[ith_row * self.n_cols + jth_col] = val

    def apply(self, func):
        return Matrix([func(a) for a in self.values], self.n_rows, self.n_cols)

    def elementwise(self, other, func):
        assert self.n_rows == other.n_rows, f"n_rows mismatch! {self.n_rows=}, {other.n_rows=}"
        assert self.n_cols == other.n_cols, "n_cols mismatch!"
        return Matrix(
            [func(a, b) for a, b in zip(self.values, other.values)],
            self.n_rows,
            self.n_cols,
        )

    def __add__(self, other):
        return self.elementwise(other, lambda a, b: a + b)

    def __mul__(self, other):
        return self.elementwise(other, lambda a, b: a * b)

    def __neg__(self):
        return self.apply(lambda a: -a)

    def __sub__(self, other):
        return self + (-other)

    def __matmul__(self, other):
        assert self.n_cols == other.n_rows, f"Dimension mismatch {self.n_cols=}, {other.n_rows=}"
        out = Matrix.zeros(self.n_rows, other.n_cols)
        for i in range(self.n_rows):
            for j in range(other.n_cols):
                for k in range(self.n_cols):
                    out[i, j] += self[i, k] * other[k, j]
        return out

    def rowsum(self):
        out = Matrix.zeros(self.n_rows, 1)
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                out[i, 0] += self[i, j]
        return out

    def rowadd(self, other):
        assert other.n_rows == 1
        assert self.n_cols == other.n_cols
        out = Matrix.zeros(self.n_rows, self.n_cols)
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                out[i, j] = self[i, j] + other[0, j]
        return out

    def sum(self):
        out = Var(0.0)
        for v in self.values:
            out += v
        return out

    def mean(self):
        return self.sum() / Var(len(self.values))


class Layer:
    def __init__(self, n_in, n_out):
        self.weights = Matrix.random(n_in, n_out, 1.0)
        self.bias = Matrix.random(1, n_out, 1.0)

    def forward(self, x):
        return (x @ self.weights).rowadd(self.bias)

    def step(self):
        for a in self.weights.values + self.bias.values:
            a.step()

    def zero_grad(self):
        for a in self.weights.values + self.bias.values:
            a.grad = 0.0


def mse_loss(a, b):
    diff = a - b
    square = diff * diff
    return square.mean()


def demo_matrix_sine(n_samples, n_nodes, n_epochs, lr, print_every):
    x_samples = Matrix.random(n_samples, 1, 2.0)
    y_samples = Matrix([Var(math.sin(v.val)) for v in x_samples.values], n_samples, 1)

    layer_1 = Layer(1, n_nodes)
    layer_2 = Layer(n_nodes, n_nodes)
    layer_3 = Layer(n_nodes, 1)

    for i in range(n_epochs):
        z1_samples = layer_1.forward(x_samples).relu()
        z2_samples = layer_2.forward(z1_samples).relu()
        y_predict = layer_3.forward(z2_samples).rowsum()
        loss = mse_loss(y_samples, y_predict)

        if loss.val > 10000:
            print("loss exploded! maybe lower learning rate")
            break

        loss.backprop(lr)
        if i % print_every == 0:
            print(f"{i=}: {loss.val=:.4e}")

        for layer in [layer_1, layer_2, layer_3]:
            layer.step()
            layer.zero_grad()

    return x_samples, y_samples, y_predict


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    random.seed(0)
    x, y, y_hat = demo_var_taylor(100, 5, 1000, 0.02, 50)

    def tolist(lst):
        return [v.val for v in lst]

    plt.figure()
    plt.scatter(tolist(x), tolist(y))
    plt.scatter(tolist(x), tolist(y_hat))
    plt.savefig("demo_var_taylor.png")

    x, y, y_hat = demo_matrix_sine(100, 6, 1000, 0.02, 50)

    def tolist(mtx):
        return [v.val for v in mtx.values]

    plt.figure()
    plt.scatter(tolist(x), tolist(y))
    plt.scatter(tolist(x), tolist(y_hat))
    plt.savefig("demo_matrix_sine.png")

