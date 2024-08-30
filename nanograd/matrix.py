from nanograd.var import Var


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
        assert self.n_rows == other.n_rows, f"{self.n_rows=} != {other.n_rows=}"
        assert self.n_cols == other.n_cols, f"{self.n_cols=} != {other.n_cols=}"
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
        assert self.n_cols == other.n_rows, f"{self.n_cols=} != {other.n_rows=}"
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

    def __repr__(self):
        s = f'Matrix [{self.n_rows}x{self.n_cols}]'
        for i in range(self.n_rows):
            s += '\n| '
            for j in range(self.n_cols):
                s += f'{self[i, j].val:+.4f} '
            s += '|'
        return s


def mse_loss(a, b):
    differ = a - b
    square = differ * differ
    return square.mean()


def demo_matrix_linear(n_samples, n_dim, n_epochs, lr, print_every):
    x_samples = Matrix.random(n_samples, n_dim, 1.0)
    a_matrix = Matrix.random(n_dim, n_dim, 1.0)
    y_samples = x_samples @ a_matrix

    model = Matrix.random(n_dim, n_dim, 1.0)

    loss_curve = []

    for epoch in range(n_epochs):
        y_predict = x_samples @ model
        loss = mse_loss(y_samples, y_predict)

        assert loss.val < 10000, "loss exploded! maybe lower learning rate"
        loss_curve.append(loss.val)

        loss.backprop(lr)
        if epoch % print_every == 0:
            print(f"{epoch=}: {loss.val=:.4e}")

        for v in model.values:
            v.update()

    return a_matrix, model, loss_curve


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import random
    random.seed(0)

    a_matrix, model, curve = demo_matrix_linear(32, 4, 100, 0.5, 10)
    print(a_matrix)
    print(model)

    plt.semilogy(curve)
    plt.savefig("output/demo_matrix_linear.png")

