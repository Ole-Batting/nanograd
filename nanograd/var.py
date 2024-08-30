
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

    def update(self):
        self.val -= self.grad
        self.grad = 0.0

    def __repr__(self):
        return f"Var({self.val:.4f}, grad={self.grad:.4e})"

    @staticmethod
    def random(scale):
        return Var(2 * scale * (random.random() - 0.5))


def demo_var_taylor(n_samples, n_coefs, n_epochs, lr, print_every):
    x_samples = [Var.random(3.14) for _ in range(n_samples)]
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

        if loss.val > 100000:
            print("loss exploded! maybe lower learning rate")
            break

        loss.backprop(lr)
        if epoch % print_every == 0:
            print(f"{epoch=:4d}: {loss.val=:.4e}")

        for c in c_params:
            c.update()

    print([f"{c.val:.4f}" for c in c_params])

    return x_samples, y_samples, y_predict


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    random.seed(0)
    x, y, y_hat = demo_var_taylor(100, 5, 4000, 0.001, 200)

    def tolist(lst):
        return [v.val for v in lst]

    plt.figure()
    plt.scatter(tolist(x), tolist(y), marker='1')
    plt.scatter(tolist(x), tolist(y_hat), marker='2')
    plt.savefig("output/demo_var_taylor.png")
