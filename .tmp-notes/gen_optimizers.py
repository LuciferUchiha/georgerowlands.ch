import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.edgecolor": "#888888",
    "axes.linewidth": 0.8,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

OUT = "assets/images/ml/"
C_GD = "#e4572e"      # orange-red
C_MOM = "#2e6fe4"     # blue
C_NEST = "#1ca35a"    # green
C_SGD = "#8e44ad"     # purple
C_OPT = "#111111"

# ill-conditioned quadratic f = 0.5 (a x^2 + b y^2)
A, B = 8.0, 1.0
def grad(p):
    return np.array([A * p[0], B * p[1]])
def f(X, Y):
    return 0.5 * (A * X**2 + B * Y**2)

def contour(ax, xlim=(-5, 5), ylim=(-4, 4)):
    xs = np.linspace(*xlim, 400)
    ys = np.linspace(*ylim, 400)
    X, Y = np.meshgrid(xs, ys)
    Z = f(X, Y)
    levels = np.linspace(0.5, Z.max(), 12)
    ax.contour(X, Y, Z, levels=levels, colors="#b8c4d0", linewidths=0.9)
    ax.plot(0, 0, marker="*", color=C_OPT, markersize=16, zorder=5)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")

def run_gd(p0, lr, steps=40):
    p = np.array(p0, float); path = [p.copy()]
    for _ in range(steps):
        p = p - lr * grad(p); path.append(p.copy())
    return np.array(path)

def run_momentum(p0, lr, beta, steps=40):
    p = np.array(p0, float); v = np.zeros(2); path = [p.copy()]
    for _ in range(steps):
        v = beta * v - lr * grad(p); p = p + v; path.append(p.copy())
    return np.array(path)

def run_nesterov(p0, lr, beta, steps=40):
    p = np.array(p0, float); pprev = p.copy(); path = [p.copy()]
    for _ in range(steps):
        look = p + beta * (p - pprev)
        pnew = look - lr * grad(look)
        pprev = p; p = pnew; path.append(p.copy())
    return np.array(path)

def run_rmsprop(p0, lr, rho=0.9, eps=1e-8, steps=40):
    p = np.array(p0, float); G = np.zeros(2); path = [p.copy()]
    for _ in range(steps):
        g = grad(p); G = rho * G + (1 - rho) * g * g
        p = p - lr * g / (np.sqrt(G) + eps); path.append(p.copy())
    return np.array(path)

def run_adam(p0, lr, b1=0.9, b2=0.999, eps=1e-8, steps=40):
    p = np.array(p0, float); m = np.zeros(2); v = np.zeros(2); path = [p.copy()]
    for t in range(1, steps + 1):
        g = grad(p)
        m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1**t); vh = v / (1 - b2**t)
        p = p - lr * mh / (np.sqrt(vh) + eps); path.append(p.copy())
    return np.array(path)

def draw_path(ax, path, color, label):
    ax.plot(path[:, 0], path[:, 1], "-o", color=color, markersize=3.5,
            linewidth=1.6, label=label, zorder=4)
    ax.plot(path[0, 0], path[0, 1], marker="o", color=C_OPT, markersize=7, zorder=6)

def arrow(ax, tail, tip, color, ls="-", lw=2.4, label=None, lpos=0.5, loff=(0, 0)):
    ax.annotate("", xy=tip, xytext=tail,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, linestyle=ls,
                                shrinkA=0, shrinkB=0, mutation_scale=18), zorder=5)
    if label:
        mx = tail[0] + lpos * (tip[0] - tail[0]) + loff[0]
        my = tail[1] + lpos * (tip[1] - tail[1]) + loff[1]
        ax.text(mx, my, label, color=color, fontsize=12.5, ha="center", va="center", zorder=7)

def decomp_bowl(ax, aa=4.0, bb=1.0, xlim=(-0.7, 4.7), ylim=(-0.6, 3.1)):
    xs = np.linspace(*xlim, 400); ys = np.linspace(*ylim, 400)
    X, Y = np.meshgrid(xs, ys); Z = 0.5 * (aa * X**2 + bb * Y**2)
    ax.contour(X, Y, Z, levels=np.linspace(0.4, Z.max(), 10), colors="#d3dae2", linewidths=0.8)
    ax.plot(0, 0, marker="*", color=C_OPT, markersize=15, zorder=6)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
def gbowl(p, aa=4.0, bb=1.0):
    return np.array([aa * p[0], bb * p[1]])

# ---- Figure 1: GD vs Momentum on an ill-conditioned bowl ----
start = [-4.3, 3.4]
fig, ax = plt.subplots(figsize=(6.4, 4.6))
contour(ax)
draw_path(ax, run_gd(start, lr=0.235, steps=34), C_GD, "gradient descent")
draw_path(ax, run_momentum(start, lr=0.022, beta=0.82, steps=34), C_MOM, "momentum")
ax.set_title("Momentum damps the zigzag")
ax.legend(loc="upper right", framealpha=0.95, fontsize=11)
fig.savefig(OUT + "optimizerMomentum.png"); plt.close(fig)

# ---- Figure 2: Momentum vs Nesterov ----
fig, ax = plt.subplots(figsize=(6.4, 4.6))
contour(ax)
draw_path(ax, run_momentum(start, lr=0.03, beta=0.9, steps=34), C_MOM, "momentum")
draw_path(ax, run_nesterov(start, lr=0.03, beta=0.9, steps=34), C_NEST, "Nesterov")
ax.set_title("Nesterov looks ahead and overshoots less")
ax.legend(loc="upper right", framealpha=0.95, fontsize=11)
fig.savefig(OUT + "optimizerNesterov.png"); plt.close(fig)

# ---- Figure 2b: Adaptive methods (per-coordinate rescaling) ----
fig, ax = plt.subplots(figsize=(6.4, 4.6))
contour(ax)
draw_path(ax, run_gd(start, lr=0.235, steps=34), C_GD, "gradient descent")
draw_path(ax, run_rmsprop(start, lr=0.30, steps=34), "#e08e0b", "RMSProp")
draw_path(ax, run_adam(start, lr=0.30, steps=34), C_MOM, "Adam")
ax.set_title("Adaptive methods rescale each coordinate")
ax.legend(loc="upper right", framealpha=0.95, fontsize=11)
fig.savefig(OUT + "optimizerAdaptive.png"); plt.close(fig)

# ---- Figure 2c: all methods on one bowl ----
fig, ax = plt.subplots(figsize=(6.8, 4.8))
contour(ax)
draw_path(ax, run_gd(start, lr=0.235, steps=34), C_GD, "gradient descent")
draw_path(ax, run_momentum(start, lr=0.022, beta=0.82, steps=34), "#e08e0b", "momentum")
draw_path(ax, run_nesterov(start, lr=0.03, beta=0.9, steps=34), C_NEST, "Nesterov")
draw_path(ax, run_adam(start, lr=0.30, steps=34), C_MOM, "Adam")
ax.set_title("Optimizers compared on the same bowl")
ax.legend(loc="upper right", framealpha=0.95, fontsize=10)
fig.savefig(OUT + "optimizerComparison.png"); plt.close(fig)

# ---- Decomposition A: heavy ball update = gradient step + momentum ----
fig, ax = plt.subplots(figsize=(6.6, 4.8))
decomp_bowl(ax)
tkm1 = np.array([4.1, 2.3]); tk = np.array([2.6, 1.8]); eta = 0.15; beta = 0.85
g = -eta * gbowl(tk); mom = beta * (tk - tkm1); tk1 = tk + g + mom
arrow(ax, tkm1, tk, "#b0b0b0", lw=1.8)                                   # previous step
arrow(ax, tk, tk + g, C_GD, label=r"$-\eta\nabla\mathcal{L}(\theta_k)$", loff=(0.0, -0.24))
arrow(ax, tk + g, tk + g + mom, "#7a7a7a", ls="--", label=r"$\beta(\theta_k-\theta_{k-1})$", loff=(0.05, -0.24))
arrow(ax, tk, tk1, C_MOM, lw=3.0, label="actual step", loff=(-0.35, 0.18))
for p, nm, dxy in [(tkm1, r"$\theta_{k-1}$", (0.14, 0.05)), (tk, r"$\theta_k$", (0.16, 0.02)), (tk1, r"$\theta_{k+1}$", (-0.05, -0.28))]:
    ax.plot(*p, marker="o", color=C_OPT, markersize=6, zorder=6); ax.text(p[0] + dxy[0], p[1] + dxy[1], nm, fontsize=12.5)
ax.set_title("Heavy ball: gradient step plus momentum")
fig.savefig(OUT + "optimizerHeavyBall.png"); plt.close(fig)

# ---- Decomposition B: Nesterov takes the gradient at the look-ahead point ----
fig, ax = plt.subplots(figsize=(6.8, 4.8))
decomp_bowl(ax, xlim=(-1.6, 4.2), ylim=(-0.5, 2.2))
nkm1 = np.array([3.6, 1.15]); nk = np.array([1.6, 0.95]); nbeta = 0.9; neta = 0.22
nmom = nbeta * (nk - nkm1); look = nk + nmom            # momentum overshoots past the valley
gl = -neta * gbowl(look); nk1 = look + gl               # gradient at look-ahead brakes it back
arrow(ax, nkm1, nk, "#b0b0b0", lw=1.8)
arrow(ax, nk, look, "#7a7a7a", ls="--", label=r"$\beta(\theta_k-\theta_{k-1})$", loff=(0.1, 0.42))
arrow(ax, look, nk1, C_GD, lw=2.6, label=r"$-\eta\nabla\mathcal{L}(\theta')$", loff=(-0.05, -0.42))
arrow(ax, nk, nk1, C_NEST, lw=3.0, label="actual step", loff=(0.45, -0.2))
for p, nm, dxy in [(nk, r"$\theta_k$", (0.16, 0.06)), (look, r"$\theta'$ look-ahead", (-0.15, 0.2)), (nk1, r"$\theta_{k+1}$", (-0.28, -0.02))]:
    ax.plot(*p, marker="o", color=C_OPT, markersize=6, zorder=6); ax.text(p[0] + dxy[0], p[1] + dxy[1], nm, fontsize=12.5, ha="center")
ax.set_title("Nesterov: gradient taken at the look-ahead point")
fig.savefig(OUT + "optimizerNesterovDecomp.png"); plt.close(fig)

# ---- Figure 3: learning rate (1D parabola) ----
fig, ax = plt.subplots(figsize=(6.8, 4.6))
xs = np.linspace(-3.2, 3.2, 400)
ax.plot(xs, xs**2, color="#b8c4d0", linewidth=1.8, zorder=1)
def gd1d(x0, lr, steps):
    x = x0; xs_ = [x]
    for _ in range(steps):
        x = x - lr * 2 * x; xs_.append(x)
    return np.array(xs_)
runs = [(0.04, C_MOM, "too small", 16), (0.4, C_NEST, "good", 12), (1.04, C_GD, "too large", 6)]
for lr, color, lab, steps in runs:
    xp = gd1d(2.6, lr, steps)
    ax.plot(xp, xp**2, "-o", color=color, markersize=5, linewidth=1.5, label=f"{lab}  $\\eta$={lr}", zorder=3)
ax.plot(0, 0, marker="*", color=C_OPT, markersize=16, zorder=5)
ax.set_title("The learning rate $\\eta$")
ax.set_xlabel("$\\theta$"); ax.set_ylabel("$\\mathcal{L}(\\theta)$")
ax.set_xlim(-3.3, 3.3); ax.set_ylim(0, 9.5)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), framealpha=0.95, fontsize=11, ncol=3)
ax.grid(True, linestyle="--", alpha=0.3)
fig.savefig(OUT + "optimizerLearningRate.png"); plt.close(fig)

# ---- Figure 4: saddle point, GD vs noisy SGD ----
def grad_saddle(p):
    return np.array([2 * p[0], -2 * p[1]])
fig, ax = plt.subplots(figsize=(6.4, 4.8))
xs = np.linspace(-2.2, 2.2, 400); ys = np.linspace(-2.2, 2.2, 400)
X, Y = np.meshgrid(xs, ys); Z = X**2 - Y**2
ax.contour(X, Y, Z, levels=np.linspace(-4, 4, 17), colors="#b8c4d0", linewidths=0.9)
ax.plot(0, 0, marker="X", color=C_OPT, markersize=12, zorder=6)
ax.annotate("saddle", (0, 0), textcoords="offset points", xytext=(10, 12), fontsize=11)
# plain GD: starts exactly on the stable axis (y=0), slides into the saddle and stalls there
p = np.array([2.0, 0.0]); path = [p.copy()]
for _ in range(50):
    p = p - 0.12 * grad_saddle(p); path.append(p.copy())
path = np.array(path)
ax.plot(path[:, 0], path[:, 1], "-o", color=C_GD, markersize=3, linewidth=1.6, label="gradient descent (stalls)", zorder=4)
ax.annotate("stalls here", (path[-1, 0], path[-1, 1]), textcoords="offset points", xytext=(4, -22), fontsize=11, color=C_GD)
# noisy SGD gets kicked off the unstable axis and escapes downhill
rng = np.random.default_rng(2)
p = np.array([2.0, 0.0]); path = [p.copy()]
for _ in range(50):
    g = grad_saddle(p) + rng.normal(0, 0.3, 2)
    p = p - 0.12 * g; path.append(p.copy())
path = np.array(path)
ax.plot(path[:, 0], path[:, 1], "-o", color=C_SGD, markersize=3, linewidth=1.6, label="noisy SGD (escapes)", zorder=4)
ax.plot(2.0, 0.0, marker="o", color=C_OPT, markersize=7, zorder=6)
ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2)
ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
ax.set_title("Noise helps escape a saddle point")
ax.legend(loc="upper right", framealpha=0.95, fontsize=11)
fig.savefig(OUT + "optimizerSaddle.png"); plt.close(fig)

print("done: optimizerMomentum, optimizerNesterov, optimizerLearningRate, optimizerSaddle")
