def rk4_system(
    t0: float,
    x0: Vector,
    h: float,
    steps: int,
    f: Callable[[float, Vector], Vector],
) -> tuple[Vector, Vector]:
    t = np.linspace(t0, t0 + h * steps, steps + 1)
    x = np.empty((steps + 1, len(x0)))
    x[0] = x0

    for i in range(steps):
        x_i = x[i]
        t_i = t[i]

        k1 = f(t_i, x_i)
        k2 = f(t_i + h / 2, x_i + (h / 2) * k1)
        k3 = f(t_i + h / 2, x_i + (h / 2) * k2)
        k4 = f(t_i + h, x_i + h * k3)

        x_next = x_i + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        x[i + 1] = x_next

    return (t, x)
