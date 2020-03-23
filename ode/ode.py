# ODE solver for any order
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from copy import deepcopy

# reading user friendly input: see parsers

# evolution of system
class ODE:
    def __init__(self, t0, x0, dxExpr=None):
        self.t = t0 # t :: R
        self.x = x0 # x :: R^n
        #self.dx = ... # dx :: R -> R^n -> R^n
        pass

    #def evolveIterate(self, dt, iters, method="euler"):
    #    pass

    # returns None & update state
    def evolve(self, dt, method="euler"):
        if method == "euler": self._euler(dt)
        elif method == "rk4": self._rk4(dt)

    def _euler(self, dt):
        self.x += self.dx(self.t, self.x) * dt
        self.t += dt

    def _rk4(self, dt):
        c1 = dt * self.dx(self.t, self.x)
        c2 = dt * self.dx(self.t + dt / 2, self.x + c1/2)
        c3 = dt * self.dx(self.t + dt / 2, self.x + c2/2)
        c4 = dt * self.dx(self.t + dt, self.x + c3)
        self.x += (c1 + 2*c2 + 2*c3 + c4) / 6. # how is this the simpson's rule?
        self.t += dt

# error analysis
# idea: track the 'energy of the system over time'

# plotting
# idea: add animations

def test():
    x0 = 1.
    ode = ODE(0., np.array([x0]))
    def dx(t, x):
        # solution is x(t)=c*exp(t)-t-1
        # for x0=1, c=2
        return np.array([x[0] + t])
    ode.dx = dx
    iters = 10
    dt = 0.5
    method = "rk4"
    res = [deepcopy(ode)]
    for _ in range(iters):
        ode.evolve(dt, method=method)
        res.append(deepcopy(ode))

    trueTs = np.linspace(0, dt*iters, 1000)
    trueXs = 2*np.exp(trueTs)-trueTs-1
    plt.plot(trueTs, trueXs, '-', linewidth=0.5, color="blue")

    xs = [o.x[0] for o in res]
    ts = [o.t for o in res]
    plt.plot(ts, xs, '.--', markersize=0.7, linewidth=0.5, color="purple")

    legend1 = "True solution: " + r"$x(t)=2e^t-t-1$"
    legend2 = "Solution by {} with ".format(method) + r"$\Delta t=$"+"${}$".format(dt)
    plt.legend([legend1, legend2])
    plt.xlabel("$t$")
    plt.ylabel("$x(t)$")
    plt.title("Numerical Solution of " + r"$\frac{dx}{dt}=x+t$")
    #plt.savefig("testEuler.svg", format='svg')
    plt.show()

def lotkaVolterra():
    x0 = 2.
    y0 = 1.
    ode = ODE(0, np.array([x0, y0]))
    """
    Things to try:
    setting c to 0, an immortal predator
    rk4 vs euler
    """
    def lotkaVolterra(a=2, b=1, c=1, d=2):
        assert a >= 0 and b >= 0 and c >= 0 and d >= 0
        def dx(t, x):
            return np.array([
            a*x[0]-b*x[0]*x[1],
            d*x[0]*x[1]-c*x[1],
            ])
        return dx
    a = 2; b = 1; c = 0; d = 2
    ode.dx = lotkaVolterra(a, b, c, d)
    iters = 500
    dt = 0.01
    method = "euler"
    res = [deepcopy(ode)]
    for _ in range(iters):
        ode.evolve(dt, method=method)
        res.append(deepcopy(ode))
    preys = [o.x[0] for o in res]
    preds = [o.x[1] for o in res]
    ts = [o.t for o in res]

    fig = plt.figure(figsize=(6, 8))
    #fig = plt.gcf()
    plt.subplot("211")
    plt.tight_layout(pad=5.)
    plt.plot(preys, preds, '.-', markersize=0.7, linewidth=0.2)
    plt.title("Lotka-Volterra Phase Portrait")
    plt.xlabel("Prey")
    plt.ylabel("Predator")
    textFactor = 1000
    for j in range(round(dt * iters)+1):
        i = round(float(j) / dt)
        plt.text(res[i].x[0], res[i].x[1], "$t = {:.2f}$".format(res[i].t))
    plt.subplot("212")
    plt.plot(ts, preys, '.-', markersize=0.7, linewidth=0.2, color='green')
    plt.plot(ts, preds, '.-', markersize=0.7, linewidth=0.2, color='red')
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.text(0, x0, "Prey", color='green', horizontalalignment='right')
    plt.text(0, y0, "Predator", color='red', horizontalalignment='right')
    fig.text(0.3, 0.48, r"$\frac{dx}{dt}=\alpha x-\beta xy$"
                        "\n"
                        r"$\frac{dy}{dt}=\delta xy-\gamma y$", fontsize=12,
                        horizontalalignment='center',
                        verticalalignment='center')
    fig.text(0.65, 0.48, r"$\alpha=$" + "${:.2f}$".format(a) +
                        "\n"
                        r"$\gamma=$" + "${:.2f}$".format(c), fontsize=12,
                        horizontalalignment='center',
                        verticalalignment='center')
    fig.text(0.8, 0.48, r"$\beta=$" + "${:.2f}$".format(b) +
                        "\n"
                        r"$\delta=$" + "${:.2f}$".format(d), fontsize=12,
                        horizontalalignment='center',
                        verticalalignment='center')
    fig.text(0.1, 0.48, r"$\Delta t=$" + "${}$".format(dt) +
                        "\n" +
                        "Method: {}".format(method), fontsize=12,
                        horizontalalignment='center',
                        verticalalignment='center')
    #plt.savefig("LV10.svg", format='svg')
    plt.show()

def twoBodyEarthSun():
    # state for each body is [x, y] for position, [u, v] for speed
    # total state is [x1,y1,u1,v1,x2,y2,u2,v2]

    m1 = 2.00 * 1e30; m2 = 5.97 * 1e24; G = 6.67 * 1e-11
    x0 = np.array([0., 0., 0., 0., 1.50 * 1e11, 0., 0., 29.8 * 1e3])
    # returns gravitational force from first object on second according
    # Newton's law of gravitation
    def gForce(x):
        r = x[0:2] - x[4:6]
        return (G * m1 * m2 / ((r[0]**2 + r[1]**2) ** (1.5))) * r

    def dx(t, x):
        r = gForce(x)
        return np.array([
            x[2],
            x[3],
            -r[0]/m1,
            -r[1]/m1,
            x[6],
            x[7],
            r[0]/m2,
            r[1]/m2,
        ])

    ode = ODE(0, x0)
    ode.dx = dx
    year = 3.154 * 1e7
    dt = 1e4
    iters = round(year / dt) + 1
    method = "euler"
    res = [deepcopy(ode)]
    for _ in range(iters):
        ode.evolve(dt, method=method)
        res.append(deepcopy(ode))
    x1s = [o.x[0] for o in res]
    y1s = [o.x[1] for o in res]
    x2s = [o.x[4] for o in res]
    y2s = [o.x[5] for o in res]
    ts = [o.t for o in res]
    plt.plot(x1s, y1s, '.-', markersize=0.7, linewidth=0.4, color='orange')
    plt.plot(x2s, y2s, '.-', markersize=0.7, linewidth=0.2, color='blue')
    #plt.xlim(-x0[4], x0[4])
    #plt.ylim(-x0[4], x0[4])
    legend1 = "Sun,   method: {}".format(method)
    legend2 = r"Earth, $\Delta t=$" + "${}s$".format(dt)
    plt.legend([legend1, legend2], loc="center left")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.title("Motion of the Earth Around the Sun for One Year")
    #plt.savefig("EarthSun2.svg", format='svg')
    plt.show()

def twoBody():
    # state for each body is [x, y] for position, [u, v] for speed
    # total state is [x1,y1,u1,v1,x2,y2,u2,v2]

    m1 = 1; m2 = 1; G = 1
    x0 = np.array([0., 0., 0., -1., 1, 0., 0.2, 0.2])
    # returns gravitational force from first object on second according
    # Newton's law of gravitation
    def gForce(x):
        r = x[0:2] - x[4:6]
        return (G * m1 * m2 / ((r[0]**2 + r[1]**2) ** (1.5))) * r

    def dx(t, x):
        r = gForce(x)
        return np.array([
            x[2],
            x[3],
            -r[0]/m1,
            -r[1]/m1,
            x[6],
            x[7],
            r[0]/m2,
            r[1]/m2,
        ])

    ode = ODE(0, x0)
    ode.dx = dx
    dt = 0.001
    iters = 6000
    method = "rk4"
    res = [deepcopy(ode)]
    for _ in range(iters):
        ode.evolve(dt, method=method)
        res.append(deepcopy(ode))
    x1s = [o.x[0] for o in res]
    y1s = [o.x[1] for o in res]
    x2s = [o.x[4] for o in res]
    y2s = [o.x[5] for o in res]
    ts = [o.t for o in res]
    plt.plot(x1s, y1s, '.-', markersize=0.7, linewidth=0.4, color='orange')
    plt.plot(x2s, y2s, '.-', markersize=0.7, linewidth=0.2, color='blue')

    X = res[0].x
    plt.arrow(X[0], X[1], X[2], X[3], color="orange", head_width=0.05)
    plt.arrow(X[4], X[5], X[6], X[7], color="blue", head_width=0.05)
    plt.text(X[0], X[1], r"$t=0$" +
                        "\n"
                        "$position=({},{})$".format(X[0], X[1])+
                        "\n" +
                        "$velocity=({},{})$".format(X[2], X[3]), fontsize=12,
                        color = "orange",
                        horizontalalignment='right',
                        verticalalignment='top')
    plt.text(X[4], X[6], r"$t=0$" +
                        "\n"
                        "$position=({},{})$".format(X[4], X[5]) +
                        "\n" +
                        "$velocity=({},{})$".format(X[6], X[7]), fontsize=12,
                        color = "blue",
                        horizontalalignment='left',
                        verticalalignment='bottom')
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    legend1 = "Mass: {}, method: {}".format(m1, method)
    legend2 = "Mass: {}, ".format(m2) + r"$\Delta t=$" + "${}$".format(dt)
    plt.legend([legend1, legend2])
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.title(r"Two Body Motion for $t \in " + "[{}, {}]$".format(res[0].t, iters*dt))
    #plt.savefig("twoBody5.svg", format='svg')
    plt.show()

if __name__ == "__main__":
    lotkaVolterra()
