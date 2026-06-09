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
C_GRID = "#b8c4d0"

# =====================================================================
# Figure 1: scalar gradient field + hyperbola of minima + saddle
# loss l(u,v) = 1/2 (a - uv)^2,  a = 1
# negative gradient (descent flow): (v(a-uv), u(a-uv))
# =====================================================================
a = 1.0
lim = 2.0
xs = np.linspace(-lim, lim, 400)
ys = np.linspace(-lim, lim, 400)
U, V = np.meshgrid(xs, ys)
Fu = V * (a - U * V)
Fv = U * (a - U * V)

fig, ax = plt.subplots(figsize=(6.0, 5.4))
speed = np.sqrt(Fu**2 + Fv**2)
ax.streamplot(U, V, Fu, Fv, color="#cdd6e0", density=1.1, linewidth=0.8,
              arrowsize=0.8, zorder=1)
# hyperbola uv = a, two branches
t = np.linspace(0.5 / lim, lim, 200)
ax.plot(t, a / t, color=C_GD, linewidth=2.4, zorder=3, label=r"minima $uv=a$")
ax.plot(-t, -a / t, color=C_GD, linewidth=2.4, zorder=3)
# saddle at origin
ax.plot(0, 0, marker="X", color=C_OPT, markersize=12, zorder=5)
ax.annotate("saddle", (0, 0), textcoords="offset points", xytext=(10, 10), fontsize=12)
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_xlabel("$u$"); ax.set_ylabel("$v$")
ax.set_aspect("equal")
ax.set_title("Gradient field of $\\frac{1}{2}(a-uv)^2$")
ax.legend(loc="upper right", framealpha=0.95, fontsize=11)
fig.savefig(OUT + "matrixCompletionScalarField.png"); plt.close(fig)

# =====================================================================
# Figure 2: gradient-flow ODE x(t) for several initializations
# x(t) = a c e^{2at} / (c e^{2at} + a - c),  a = 1
# =====================================================================
def x_of_t(t, c, a=1.0):
    e = np.exp(2 * a * t)
    return a * c * e / (c * e + a - c)

tt = np.linspace(0, 4.0, 500)
fig, ax = plt.subplots(figsize=(6.6, 4.6))
ax.axhline(a, color=C_OPT, linestyle="--", linewidth=1.2, zorder=1)
ax.text(0.05, a + 0.03, "target $x=a$", fontsize=11, color=C_OPT)
ax.axhline(0, color="#999999", linestyle=":", linewidth=1.0, zorder=1)
ax.text(3.0, 0.03, "saddle $x=0$", fontsize=11, color="#666666")
for c, color in [(0.01, C_MOM), (0.25, C_NEST), (0.81, C_GD)]:
    ax.plot(tt, x_of_t(tt, c), color=color, linewidth=2.2,
            label=f"$x(0)={c}$", zorder=3)
ax.set_xlim(0, 4.0); ax.set_ylim(-0.05, 1.15)
ax.set_xlabel("time $t$"); ax.set_ylabel(r"$x(t)=uv$")
ax.set_title("Gradient flow escapes the saddle")
ax.legend(loc="center right", framealpha=0.95, fontsize=11)
ax.grid(True, linestyle="--", alpha=0.3)
fig.savefig(OUT + "matrixCompletionScalarODE.png"); plt.close(fig)

# =====================================================================
# Figure 3: convex envelope of the rank-like step function
# on [-1,1]: g(x)=0 at x=0, 1 otherwise; convex envelope is |x|
# =====================================================================
fig, ax = plt.subplots(figsize=(6.4, 4.4))
xs = np.linspace(-1, 1, 400)
# rank-like step (the "0-norm" of a scalar)
ax.hlines(1, -1, -0.0001, color="#888888", linewidth=2.2, zorder=2)
ax.hlines(1, 0.0001, 1, color="#888888", linewidth=2.2, zorder=2,
          label=r"rank-like step $\|\sigma\|_0$")
ax.plot(0, 1, marker="o", markerfacecolor="white", markeredgecolor="#888888",
        markersize=8, zorder=3)
ax.plot(0, 0, marker="o", color="#888888", markersize=8, zorder=3)
# convex envelope |x|  (the "1-norm")
ax.plot(xs, np.abs(xs), color=C_MOM, linewidth=2.6,
        label=r"convex envelope $|\sigma|=\|\sigma\|_1$", zorder=4)
ax.set_xlim(-1.05, 1.05); ax.set_ylim(-0.1, 1.25)
ax.set_xlabel(r"singular value $\sigma$")
ax.set_title("Nuclear norm is the convex envelope of rank")
ax.legend(loc="upper center", framealpha=0.95, fontsize=11)
ax.grid(True, linestyle="--", alpha=0.3)
fig.savefig(OUT + "matrixCompletionConvexEnvelope.png"); plt.close(fig)

# =====================================================================
# Figure 4: projection method schematic
# ambient matrix space, non-convex rank-k set, gradient step + SVD projection
# =====================================================================
fig, ax = plt.subplots(figsize=(6.8, 4.8))
# ambient space box
ax.add_patch(plt.Rectangle((0, 0), 10, 7, fill=False, edgecolor="#b8c4d0",
                           linewidth=1.4))
ax.text(0.3, 6.5, "space of all matrices", fontsize=12, color="#7a7a7a")
# non-convex rank-k set as a wavy curve
cx = np.linspace(0.6, 9.4, 300)
cy = 2.4 + 1.1 * np.sin(0.7 * cx) + 0.15 * cx
ax.plot(cx, cy, color=C_NEST, linewidth=2.6, zorder=2)
ax.text(6.7, 1.7, "rank-$k$ matrices", fontsize=12.5, color=C_NEST)

def on_curve(x):
    return 2.4 + 1.1 * np.sin(0.7 * x) + 0.15 * x

def arrow(tail, tip, color, lw=2.2, label=None, loff=(0, 0)):
    ax.annotate("", xy=tip, xytext=tail,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                shrinkA=0, shrinkB=0, mutation_scale=16), zorder=4)
    if label:
        ax.text((tail[0] + tip[0]) / 2 + loff[0], (tail[1] + tip[1]) / 2 + loff[1],
                label, color=color, fontsize=11.5, ha="center", va="center", zorder=6)

# iterate sequence: on curve -> gradient step off -> project back
p0 = np.array([2.2, on_curve(2.2)])
g0 = np.array([3.6, on_curve(2.2) + 2.0])     # gradient step leaves the set
p1 = np.array([4.2, on_curve(4.2)])           # SVD projection back onto set
g1 = np.array([5.7, on_curve(4.2) + 1.9])
p2 = np.array([6.4, on_curve(6.4)])
arrow(p0, g0, C_GD, label="gradient\nstep", loff=(-0.95, 0.1))
arrow(g0, p1, "#7a7a7a", label="SVD\nprojection", loff=(0.95, 0.25))
arrow(p1, g1, C_GD)
arrow(g1, p2, "#7a7a7a")
for p, nm, dxy in [(p0, r"$\mathbf{A}^t$", (-0.1, -0.45)),
                   (p1, r"$\mathbf{A}^{t+1}$", (0.1, -0.5)),
                   (p2, r"$\mathbf{A}^{t+2}$", (0.2, -0.5))]:
    ax.plot(*p, marker="o", color=C_OPT, markersize=7, zorder=6)
    ax.text(p[0] + dxy[0], p[1] + dxy[1], nm, fontsize=12, ha="center")
for p in [g0, g1]:
    ax.plot(*p, marker="o", markerfacecolor="white", markeredgecolor=C_GD,
            markersize=6, zorder=6)
ax.set_xlim(-0.3, 10.3); ax.set_ylim(-0.3, 7.3)
ax.set_xticks([]); ax.set_yticks([])
ax.set_aspect("equal")
ax.set_title("Singular value projection")
fig.savefig(OUT + "matrixCompletionProjection.png"); plt.close(fig)

# =====================================================================
# Figure 5: ALS vs gradient descent convergence on a small completion problem
# =====================================================================
rng = np.random.default_rng(0)
n, m, k = 40, 30, 3
Utrue = rng.normal(size=(n, k)); Vtrue = rng.normal(size=(m, k))
A = Utrue @ Vtrue.T
Omega = (rng.random((n, m)) < 0.6).astype(float)   # 60% observed
lam = 0.1

def objective(U, V):
    R = Omega * (A - U @ V.T)
    return 0.5 * np.sum(R**2) + 0.5 * lam * (np.sum(U**2) + np.sum(V**2))

ITERS = 60
# --- ALS ---
U = rng.normal(size=(n, k)) * 0.1
V = rng.normal(size=(m, k)) * 0.1
als = [objective(U, V)]
for _ in range(ITERS):
    for j in range(m):
        w = Omega[:, j]
        Mat = (U * w[:, None]).T @ U + lam * np.eye(k)
        V[j] = np.linalg.solve(Mat, (U * w[:, None]).T @ A[:, j])
    for i in range(n):
        w = Omega[i, :]
        Mat = (V * w[:, None]).T @ V + lam * np.eye(k)
        U[i] = np.linalg.solve(Mat, (V * w[:, None]).T @ A[i, :])
    als.append(objective(U, V))

# --- gradient descent on the same objective ---
U = rng.normal(size=(n, k)) * 0.1
V = rng.normal(size=(m, k)) * 0.1
eta = 0.01
gd = [objective(U, V)]
for _ in range(ITERS):
    R = Omega * (U @ V.T - A)
    gU = R @ V + lam * U
    gV = R.T @ U + lam * V
    U -= eta * gU; V -= eta * gV
    gd.append(objective(U, V))

fig, ax = plt.subplots(figsize=(6.6, 4.6))
ax.semilogy(als, "-o", color=C_MOM, markersize=4, linewidth=1.8, label="ALS")
ax.semilogy(gd, "-o", color=C_GD, markersize=4, linewidth=1.8, label="gradient descent")
ax.set_xlabel("iteration"); ax.set_ylabel("objective $\\phi$ (log scale)")
ax.set_title("ALS converges in few iterations")
ax.legend(loc="upper right", framealpha=0.95, fontsize=11)
ax.grid(True, which="both", linestyle="--", alpha=0.3)
fig.savefig(OUT + "matrixCompletionALS.png"); plt.close(fig)

print("done: ScalarField, ScalarODE, ConvexEnvelope, Projection, ALS")
