import math
from nanograd.matrix import Matrix, mse_loss
from nanograd.var import Var


class Layer:
    def __init__(self, n_in, n_out, activation):
        self.weights = Matrix.random(n_in, n_out, 1.0)
        self.bias = Matrix.random(1, n_out, 1.0)
        self.activation = activation

    def forward(self, x):
        z = x @ self.weights
        z = z.rowadd(self.bias)
        z = self.activation(z)
        return z

    def update(self):
        for a in self.weights.values + self.bias.values:
            a.update()


def identity(x: Matrix):
    return x


def _relu(x: Var):
    return x if x.val > 0.0 else Var(0.0)


def relu(x: Matrix):
    return x.apply(_relu)


def _tanh(x: Var):
    return Var(math.tanh(x.val), [(x, 1 - math.tanh(x.val) ** 2)])

def tanh(x: Matrix):
    return x.apply(_tanh)


def demo_layer_sine(n_samples, n_nodes, n_epochs, lr, print_every):
    x_samples = Matrix.random(n_samples, 1, 3.14)
    y_samples = x_samples.apply(lambda x: Var(math.sin(x.val)))

    layer_1 = Layer(1, n_nodes, tanh)
    layer_2 = Layer(n_nodes, n_nodes, tanh)
    layer_3 = Layer(n_nodes, 1, identity)

    for epoch in range(n_epochs):
        z_samples = layer_1.forward(x_samples)
        z_samples = layer_2.forward(z_samples)
        y_predict = layer_3.forward(z_samples)
        loss = mse_loss(y_samples, y_predict)

        assert loss.val < 10000, "loss exploded! maybe lower learning rate"

        loss.backprop(lr)
        if epoch % print_every == 0:
            print(f"{epoch=:4d}: {loss.val=:.4e}")

        for layer in [layer_1, layer_2, layer_3]:
            layer.update()

    return x_samples, y_samples, y_predict


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import random
    random.seed(0)

    x, y, y_hat = demo_layer_sine(50, 7, 1000, 0.04, 50)

    def tolist(mtx):
        return [v.val for v in mtx.values]

    plt.scatter(tolist(x), tolist(y), marker='1')
    plt.scatter(tolist(x), tolist(y_hat), marker='2')
    plt.savefig("output/demo_layer_sine.png")

