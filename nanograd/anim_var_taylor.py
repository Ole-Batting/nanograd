import math
import random
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
from nanograd.var import Var


def anim_var_taylor(n_samples, n_coefs, n_epochs, lr):
    x_samples = [Var.random(3.14) for _ in range(n_samples)]
    y_samples = [Var(math.sin(x.val)) for x in x_samples]

    c_params = [Var.random(1.0) for _ in range(n_coefs)]

    def sine_taylor(x):
        out = Var(0.0)
        for i, c in enumerate(c_params):
            out += c * x ** (2 * i + 1)
        return out

    y_predict = [Var(0.0) for _ in range(n_samples)]
    snapshots = [[param.val for param in c_params]]

    for epoch in trange(n_epochs):
        loss = Var(0.0)
        for i, (x, y) in enumerate(zip(x_samples, y_samples)):
            y_hat = sine_taylor(x)
            y_predict[i] = y_hat
            loss += (y - y_hat) ** 2
        loss /= Var(n_samples)

        if loss.val > 1000000:
            print("loss exploded! maybe lower learning rate")
            break

        loss.backprop(lr)

        for c in c_params:
            c.update()

        snapshots.append([param.val for param in c_params])

    print([f"{c.val:.4f}" for c in c_params])

    return x_samples, y_samples, snapshots

if __name__ == '__main__':
    random.seed(0)

    n_frames = 4000
    n_skip = 20
    x_samples, y_samples, snapshots = anim_var_taylor(100, 2, n_frames, 0.001)

    def tolist(lst):
        return [v.val for v in lst]

    def sine_taylor(x, c_params):
        out = np.zeros_like(x)
        for i, c in enumerate(c_params):
            out += c * x ** (2 * i + 1)
        return out

    x_space = np.linspace(-3.14, 3.14, 100)

    fig, ax = plt.subplots()
    sc = ax.scatter(tolist(x_samples), tolist(y_samples), label='Cosine')
    ln = ax.plot(x_space, sine_taylor(x_space, snapshots[0]), c='orange', label='Taylor series estimate')[0]
    ax.set(xlim=[-3.14, 3.14], ylim=[-1.1, 1.1], xlabel='input', ylabel='output')
    ax.legend()

    def update(frame):
        ln.set_ydata(sine_taylor(x_space, snapshots[(frame + 1) * n_skip]))
        return sc, ln

    ani = animation.FuncAnimation(fig=fig, func=update, frames=n_frames // n_skip, interval=20)
    ani.save("output/anim_var_taylor.mp4")

