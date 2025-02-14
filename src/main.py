from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

type Vector = np.typing.NDArray[np.floating]
type Matrix = np.typing.NDArray[np.floating]


def k_ify(k: list[float]) -> Matrix:
    """
    Takes a list of damping/elastic coefficients and generates the appropiate matrix for the system.
    """
    n = len(k)

    if n == 1:
        return np.array([k[0]])

    out = np.zeros((n, n))

    out[0][0] = k[0] + k[1]
    out[0][1] = -k[1]

    for i in range(1, n - 1):
        out[i][i - 1] = -k[i]
        out[i][i] = k[i] + k[i + 1]
        out[i][i + 1] = -k[i + 1]

    out[n - 1][n - 2] = -k[n - 1]
    out[n - 1][n - 1] = k[n - 1]

    return out


# m_mat = np.diag([5, 4, 3, 2, 1])
# c_mat = k_ify([0.4, 0.2, 0.2, 0.1, 0.05])
# k_mat = k_ify([3, 3, 3, 3, 3])

m_mat = np.diag([1])
c_mat = k_ify([0.5])
k_mat = k_ify([1])

n = len(m_mat)

t0 = 0
tN = 100
steps = 10000
h = (tN - t0) / steps

u0 = np.zeros(2 * n).transpose()


def p(t: float) -> Vector:
    out = np.zeros(n)
    out[0] = 10
    return out


M_inv = np.linalg.inv(m_mat)


def f(t: float, x: Vector) -> Vector:
    v = x[:n]
    u = x[n:]
    v_prime = M_inv @ (-c_mat @ v - k_mat @ u + p(t))
    u_prime = v
    return np.append(v_prime, u_prime)


def rk4_system(
    t0: float,
    x0: Vector,
    h: float,
    steps: int,
    f: Callable[[float, Vector], Vector],
) -> tuple[Vector, Vector]:
    t = np.linspace(t0, t0 + h * steps, steps + 1)
    x = np.empty((len(x0), steps + 1))
    x[:, 0] = x0

    for i in range(steps):
        x_i = x[:, i]
        t_i = t[i]

        k1 = f(t_i, x_i)
        k2 = f(t_i + h / 2, x_i + (h / 2) * k1)
        k3 = f(t_i + h / 2, x_i + (h / 2) * k2)
        k4 = f(t_i + h, x_i + h * k3)

        x[:, i + 1] = x_i + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return (t, x)


(t_out, x_out) = rk4_system(t0, u0, h, steps, f)
u_out = x_out[n:]

color = plt.colormaps["gist_rainbow"](np.linspace(0, 1, n))

for i, u_i in enumerate(u_out):
    plt.plot(t_out, u_i, "-", label=f"u{i+1}(t)", c=color[i])

plt.legend()
plt.show()
