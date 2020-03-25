# Successive parabolic interpolation 17/03/20
"""
TODO:
Refactor code so the function I choose to plot with is decoupled from the
animation code.

Continuing ideas:
* try printing the error on screen
* try different functions, and save some animations
* try adding a zoom with each iteration
Interesting it looks like the interpolating parabola tends to the 2nd order
taylor polynomial about the minimum

There seems to be some numerical error in the parabola minimum for small
intervals, making it fall outside of the interval quite often. Perhaps we
can log the intervals and get a log version of the algorithm which has no
numerical issues
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import sys
from pprint import pprint

def f(x):
    return np.cos(x)

def getParabola(a, b, c, fa, fb, fc):
    # returns the coefficients of fitted parabola, in lagarange form, a 3-tuple
    # of (r, s, t) where fitted(x) = r(x-b)(x-c)+s(x-c)(x-a)+t(x-a)(x-b)
    # pre: a, b, c distinct
    r = fa/((a-b)*(a-c))
    s = fb/((b-c)*(b-a))
    t = fc/((c-a)*(c-b))
    p = (a, b, c, r, s, t)
    #assert evalParabola(p, a) == fa
    #assert evalParabola(p, b) == fb
    #assert evalParabola(p, c) == fc
    return p

def getMin(parabola): # returns pair of argmin and min of lagrange form parabola
    a, b, c, r, s, t = parabola
    d = r + s + t
    if d == 0:
        raise ValueError("Parabola is a line")
    def m(x, y): return (x+y)/2
    x = (r*m(b,c)+s*m(c,a)+t*m(a,b)) / d
    return (x, evalParabola(parabola, x))

def evalParabola(parabola, x):
    (a, b, c, r, s, t) = parabola
    return r*(x-b)*(x-c) + s*(x-c)*(x-a) + t*(x-a)*(x-b)

def isBracket(a, b, c, fa, fb, fc):
    return a < b < c and fa >= fb and fc >= fb

def spi(a, b, c, fa, fb, fc, maxIter=30, tol=1e-8): # 1e-8 is from optimisation
    # pre: (a, b, c) is a bracket of f
    # returns list of brackets (6-tuple)
    state = (a, b, c, fa, fb, fc)
    iter = 0
    while (abs(c-a) > tol and iter < maxIter):
        a, b, c, fa, fb, fc = state
        assert isBracket(a, b, c, fa, fb, fc)
        p = getParabola(a, b, c, fa, fb, fc)
        z, approxfz = getMin(p)
        fz = f(z)
        yield (state, p, z, fz)
        assert a < z < c, "z is {}, approx f(z) is {}".format(z, approxfz)
        state = refine(a, b, c, z, fa, fb, fc, fz)
        iter += 1
    yield (state, None, None, None)
    #return brackets

def refine(a, b, c, z, fa, fb, fc, fz):
    # returns refined bracket and associated function values
    # pre: a < z < b, (a, b, c) a bracket of f
    if z < b:
        if fz > fb:
            return (z, b, c, fz, fb, fc)
        #elif fz == fb:
            #raise ValueError("Refinement has f(z) = f(b)") # trouble
        elif fz <= fb:
            return (a, z, b, fa, fz, fb)
    elif z == b:
        raise ValueError("Refinement has z = b") # trouble
    elif z > b:
        if fz > fb:
            return (a, b, z, fa, fb, fz)
        #elif fz == fb:
        #    raise ValueError("Refinement has f(z) = f(b)")
        elif fz <= fb:
            return (b, z, c, fb, fz, fc)
        #Nc1, Nb1, Na1, fc1, fb1, fa1 = refine(-c, -b, -a, -z, fc, fb, fa, fz)
        #return (-Na1, -Nb1, -Nc1, fa1, fb1, fc1)

if __name__ == "__main__":
    assert len(sys.argv) == 4
    a, b, c = [float(x) for x in sys.argv[1:]]
    fa = f(a); fb = f(b); fc = f(c)
    assert isBracket(a, b, c, fa, fb, fc)
    states = spi(a, b, c, fa, fb, fc) # important

    fig = plt.figure()
    ax = plt.axes(xlim=(0, 2*np.pi), ylim=(-1.5, 1.5))

    xs = np.linspace(0, 2*np.pi, 1000) # important
    ys = f(xs)
    fLine, = ax.plot(xs, ys, lw=2, color="blue")

    plt.legend(["$cos(x)$"])
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.title("Successive Parabolic Interpolation for Local Minimum")

    bracketPts, = ax.plot([], [], 'x', color="red", markersize=10)
    parabolaLine, = ax.plot([], [], '--', color="gold")
    zLine, = ax.plot([], [], '--', color="gold")
    zPt, = ax.plot([], [], 'o', color="gold")

    y = max(fa, fc)
    bracketLine, = ax.plot([0, 0], [y, y], '.-', color="red")
    bracketInfo = ax.text(0, y, '', horizontalalignment="center",
                          verticalalignment="bottom", color="red")

    iterNum = -1

    def init():
        bracketPts.set_data([], [])
        parabolaLine.set_data([], [])
        zLine.set_data([], [])
        zPt.set_data([], [])
        return bracketPts, parabolaLine, zLine, zPt, bracketLine, bracketInfo

    def animate(i):
        global iterNum
        iterNum += 1
        state, p, z, fz = next(states)
        a, b, c, fa, fb, fc = state
        bracketPts.set_data([a, b, c], [fa, fb, fc])
        parabolaLine.set_data(xs, evalParabola(p, xs))

        approxfz = evalParabola(p, z)
        zLine.set_data([z, z], [approxfz, fz])
        zPt.set_data([z], [fz])


        bracketLine.set_xdata([a, c])#, [y, y])
        bracketInfo.set_x((a+c)/2)
        bText = ("iteration: {}\n".format(iterNum) +
                r"argmin $\in " + "({:.8f}, {:.8f})$\n".format(a, c) +
                r"error $\leq" + "{:.2E}$".format(abs(c-a)))

        bracketInfo.set_text(bText)
        return bracketPts, parabolaLine, zLine, zPt, bracketLine, bracketInfo

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=30, interval=3000, blit=True)
    #anim.save('cos3.mp4', fps=1/3, extra_args=['-vcodec', 'libx264'])
    plt.show()
