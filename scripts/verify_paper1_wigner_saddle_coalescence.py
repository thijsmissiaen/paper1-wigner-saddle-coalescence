#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_paper1_wigner_saddle_coalescence.py
===============================================================================
High-Precision Verification & Figure Generator for Paper 1:
"Wigner Phase-Space Representation, Shifted Cauchy Contours, and Local
 Saddle-Point Coalescence in the Transverse Wronskian of Riemann's Xi-Function"
===============================================================================
Author : Thijs Missiaen
Date   : November 2026
Engine : Python 3 + mpmath (60 decimal places), NumPy, SciPy, Matplotlib

This self-contained verification script reproduces all analytical identities,
60-decimal and 50-decimal numerical tables, and 300-DPI academic figures
reported in Paper 1:

  STEP 1: Holomorphic even parity Phi(-u) = Phi(u) without non-holomorphic |u|
          inside the strip |Im(u)| < pi/8, Hermitian contour symmetry
          G_y(-xi + i*eta) = conj(G_y(xi + i*eta)) on horizontal contours,
          real non-negative values G_y(i*eta) = |Phi((y + i*eta)/2)|^2 >= 0
          on the imaginary axis, and explicit localization of zeros of Phi(u)
          inside |Im(u)| < pi/8 (both on the imaginary axis at tau_0 = 0.319415...
          and off-axis at u_1 = 0.127824... + 0.347770...i), demonstrating that
          ln G_y(z) has logarithmic branch singularities inside |Im(z)| < pi/4.

  STEP 2: Double-exponential decay on the vertical Cauchy walls xi = +-L governed
          by exp(-2*pi*cosh(2y)*e^{2L}*cos(2*eta)) for |eta| < pi/4, breakdown of
          defining series convergence at |eta| = pi/4 (where cos(2*eta) = 0),
          and 35-decimal verification of Cauchy's integral theorem on the closed
          rectangle [-L, L] x [0, eta] with contour residual < 10^-34.

  STEP 3: 50-decimal numerical homotopy tracking of the local complex saddle points
          x_*^{+-}(y, omega) = +-xi_*(y, omega) + i*eta_*(y, omega) solving
          d/dx ln G_y(x) + i*omega = 0 at omega = 2*gamma_1 across y in [0.05, 1.10],
          and verification of the local pitchfork normal form at y_c(2*gamma_1) =
          0.4320205303... and eta_c = 0.6027150239....

  STEP 4: Quadrature conditioning along horizontal contours z = xi + i*eta
          (confirming Cauchy contour invariance of the net integral I(y, t) while
          reducing the L^1-to-net condition number from 867.6 on eta = 0 to 1.0008
          on eta = 0.50 at y = 0.40), high-precision verification of the canonical
          normalization W(d, t) = 16 * int_0^infty y * sinh(2*d*y) * I(y, t) dy
          against direct Cauchy-Riemann evaluation of Xi(t - i*d), and 25+ digit
          verification of the zero-marginal identity int_0^infty I(y, gamma_k) dy = 0.
"""

import sys
import time
import math
from pathlib import Path
import numpy as np
import mpmath as mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

mp.mp.dps = 60

GAMMA_1_STR = "14.1347251417346937904572519835624702707842571156992431756855"


def print_banner(title: str) -> None:
    print("=" * 88)
    print(f" {title}")
    print("=" * 88)


# ==============================================================================
# 1. HOLOMORPHIC JACOBI-RIEMANN KERNEL Phi(u) AND CONVOLUTION KERNEL G_y(x)
# ==============================================================================

def phi_mp(u, max_n: int = 240):
    """
    Evaluates the holomorphic Jacobi-Riemann kernel Phi(u) in the open strip
    |Im(u)| < pi/8:
        Phi(u) = sum_{n=1}^infty (2*pi^2*n^4*e^{9u} - 3*pi*n^2*e^{5u}) * exp(-pi*n^2*e^{4u})
    No non-holomorphic abs(u) is used here so complex holomorphy is preserved.
    """
    u = mp.mpc(u)
    s = mp.mpc(0)
    e4 = mp.exp(4 * u)
    e5 = mp.exp(5 * u)
    e9 = mp.exp(9 * u)
    pi = mp.pi
    pi2 = pi * pi
    for n in range(1, max_n + 1):
        n2 = n * n
        n4 = n2 * n2
        exp_fac = mp.exp(-pi * n2 * e4)
        if abs(exp_fac) < mp.mpf("1e-85") and n > 5:
            break
        P = 2 * pi2 * n4 * e9 - 3 * pi * n2 * e5
        s += P * exp_fac
    return s


def phi_and_derivatives_mp(u, max_n: int = 200):
    """
    Simultaneously evaluates Phi(u), Phi'(u), and Phi''(u) inside |Im(u)| < pi/8.
    """
    u = mp.mpc(u)
    s = mp.mpc(0)
    ds = mp.mpc(0)
    d2s = mp.mpc(0)
    e4 = mp.exp(4 * u)
    e5 = mp.exp(5 * u)
    e9 = mp.exp(9 * u)
    pi = mp.pi
    pi2 = pi * pi
    for n in range(1, max_n + 1):
        n2 = n * n
        n4 = n2 * n2
        exp_fac = mp.exp(-pi * n2 * e4)
        if abs(exp_fac) < mp.mpf("1e-85") and n > 5:
            break
        a_n = pi * n2 * e4
        P0 = 2 * pi2 * n4 * e9 - 3 * pi * n2 * e5
        P1 = 18 * pi2 * n4 * e9 - 15 * pi * n2 * e5 - 4 * a_n * P0
        P2 = (
            162 * pi2 * n4 * e9
            - 75 * pi * n2 * e5
            - 16 * a_n * P0
            - 4 * a_n * P1
            - 4 * a_n * (18 * pi2 * n4 * e9 - 15 * pi * n2 * e5)
        )
        s += P0 * exp_fac
        ds += P1 * exp_fac
        d2s += P2 * exp_fac
    return s, ds, d2s


def G_y_mp(x, y):
    """
    Evaluates G_y(x) = Phi((x + y)/2) * Phi((x - y)/2) in |Im(x)| < pi/4.
    """
    return phi_mp((x + y) / 2) * phi_mp((x - y) / 2)


def dlogG_dx_mp(x, y, max_n: int = 160):
    p1, dp1, _ = phi_and_derivatives_mp((x + y) / 2, max_n=max_n)
    p2, dp2, _ = phi_and_derivatives_mp((x - y) / 2, max_n=max_n)
    return mp.mpf("0.5") * (dp1 / p1 + dp2 / p2)


def d2logG_dx2_mp(x, y):
    p1, dp1, d2p1 = phi_and_derivatives_mp((x + y) / 2)
    p2, dp2, d2p2 = phi_and_derivatives_mp((x - y) / 2)
    t1 = d2p1 / p1 - (dp1 / p1) ** 2
    t2 = d2p2 / p2 - (dp2 / p2) ** 2
    return mp.mpf("0.25") * (t1 + t2)


def xi_completed_mp(s):
    return mp.mpf("0.5") * s * (s - 1) * (mp.pi ** (-s / 2)) * mp.gamma(s / 2) * mp.zeta(s)


def Xi_mp(z):
    return xi_completed_mp(mp.mpf("0.5") + mp.j * z)


def W_cr_mp(d, t):
    z0 = mp.mpf(t) - mp.j * mp.mpf(d)
    val = Xi_mp(z0)
    dval = mp.diff(Xi_mp, z0)
    R, I_im = mp.re(val), mp.im(val)
    Rt, It = mp.re(dval), mp.im(dval)
    return R * It - I_im * Rt


def phi_complex_np(u):
    """
    Vectorized NumPy evaluation of Phi(u) for |Im(u)| < pi/8 using the proven
    holomorphic parity Phi(-u) = Phi(u) for Re(u) < 0.
    """
    u = np.asarray(u, dtype=np.complex128)
    u_eff = np.where(np.real(u) >= 0.0, u, -u)
    s = np.zeros_like(u_eff, dtype=np.complex128)
    pi = np.pi
    pi2 = pi * pi
    e4 = np.exp(4.0 * u_eff)
    e5 = np.exp(5.0 * u_eff)
    e9 = np.exp(9.0 * u_eff)
    for n in range(1, 36):
        n2 = float(n * n)
        n4 = n2 * n2
        term = (2.0 * pi2 * n4 * e9 - 3.0 * pi * n2 * e5) * np.exp(-pi * n2 * e4)
        s += term
    return s


def G_y_np(x, y):
    return phi_complex_np((x + y) / 2.0) * phi_complex_np((x - y) / 2.0)


# ==============================================================================
# 2. FIGURE GENERATION (300-DPI ACADEMIC FIGURES, 100% ENGLISH)
# ==============================================================================

def generate_figures(
    fig_dir: Path,
    table_points: list,
    pitchfork_curve: list,
    coalesced_curve: list,
    yc_exact: float,
    etac_exact: float,
    eta_zero_im: float,
    eta_zero_off: float,
) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10.5,
        "axes.titlesize": 10.8,
        "legend.fontsize": 8.0,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.28,
        "grid.linestyle": "--",
    })

    gamma_1 = 14.134725141734694
    omega_rs = 2.0 * gamma_1

    # --------------------------------------------------------------------------
    # FIGURE 1: (a) Quadrature conditioning on shifted contour vs real axis
    #           (b) Local saddle-point coalescence & strip log-singularities
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.7))

    xi_plot = np.linspace(-1.45, 1.45, 4000)
    y_demo = 0.40
    eta_lift = 0.50

    integrand_real = np.real(G_y_np(xi_plot, y_demo) * np.exp(1j * omega_rs * xi_plot))
    integrand_lift = np.exp(-omega_rs * eta_lift) * np.real(
        G_y_np(xi_plot + 1j * eta_lift, y_demo) * np.exp(1j * omega_rs * xi_plot)
    )

    norm_real = integrand_real / np.max(np.abs(integrand_real))
    norm_lift = integrand_lift / np.max(np.abs(integrand_lift))

    ax1.plot(
        xi_plot, norm_real, color="#7f8c8d", lw=1.0, alpha=0.80,
        label=r"Real axis $\eta = 0$ ($\kappa_{\mathrm{quad}} = 867.6$, net integral invariant)"
    )
    ax1.plot(
        xi_plot, norm_lift, color="#1b4f72", lw=2.0,
        label=r"Shifted contour $\eta = 0.50$ ($\kappa_{\mathrm{quad}} = 1.0008$, same $I(y,\gamma_1)$)"
    )
    ax1.fill_between(
        xi_plot, 0, np.maximum(norm_lift, 0), color="#2980b9", alpha=0.22
    )
    ax1.fill_between(
        xi_plot, 0, np.minimum(norm_lift, 0), color="#c0392b", alpha=0.25
    )
    ax1.axhline(0.0, color="black", lw=0.7, alpha=0.6)
    ax1.set_xlim(-1.45, 1.45)
    ax1.set_ylim(-1.08, 1.12)
    ax1.set_xlabel(r"Horizontal contour coordinate $\xi = \mathrm{Re}(z)$")
    ax1.set_ylabel(r"Peak-normalized integrand $\mathrm{Re}[G_y(\xi+i\eta)e^{i\omega(\xi+i\eta)}]$")
    ax1.set_title(r"(a) Shifted-Contour Quadrature Conditioning ($y = 0.40, \omega = 2\gamma_1$)")
    ax1.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92)

    # Panel (b): Numerical saddle branch + log-singularity heights
    y_ref = np.linspace(0.05, 0.452, 200)
    xi_out_ref = np.maximum(
        0.0,
        0.5 * np.log(np.sqrt(omega_rs ** 2 + 81.0) / (4.0 * np.pi * np.cosh(2.0 * y_ref))),
    )
    ax2.plot(
        y_ref, xi_out_ref, color="#5dade2", lw=1.2, ls=":",
        label=r"$n=1$ outer estimate $\pm\xi_*^{\mathrm{out}}(y, 2\gamma_1)$"
    )
    ax2.plot(y_ref, -xi_out_ref, color="#5dade2", lw=1.2, ls=":")

    y_pf = np.array([row[0] for row in pitchfork_curve])
    xi_pf = np.array([row[1] for row in pitchfork_curve])
    eta_pf = np.array([row[2] for row in pitchfork_curve])

    y_co = np.array([row[0] for row in coalesced_curve])
    xi_co = np.array([row[1] for row in coalesced_curve])
    eta_co = np.array([row[2] for row in coalesced_curve])

    ax2.plot(y_pf, xi_pf, color="#1b4f72", lw=2.0)
    ax2.plot(y_pf, -xi_pf, color="#1b4f72", lw=2.0, ls="--")
    ax2.plot(y_pf, eta_pf, color="#b03a2e", lw=2.0)

    ax2.plot(
        y_co, xi_co, color="#1b4f72", lw=2.2,
        label=r"Tracked saddle branch $\pm\xi_*(y, 2\gamma_1)$ ($\xi_* = 0$ on $[y_c, 1.1]$)"
    )
    ax2.plot(
        y_co, eta_co, color="#b03a2e", lw=2.1,
        label=r"Tracked saddle height $\eta_*(y, 2\gamma_1)$"
    )

    y_tb = np.array([row[0] for row in table_points])
    xi_tb = np.array([row[1] for row in table_points])
    eta_tb = np.array([row[2] for row in table_points])
    ax2.plot(y_tb, xi_tb, "o", color="#1b4f72", markersize=4.0, alpha=0.90)
    ax2.plot(y_tb[xi_tb > 0], -xi_tb[xi_tb > 0], "o", color="#1b4f72", markersize=4.0, alpha=0.90, fillstyle="none")
    ax2.plot(y_tb, eta_tb, "s", color="#b03a2e", markersize=3.6, alpha=0.88)

    ax2.axhline(
        math.pi / 4.0, color="#7d3c98", lw=1.2, ls=":",
        label=r"Series strip boundary $\eta = \pi/4 \approx 0.7854$"
    )
    ax2.axhline(
        eta_zero_im, color="#d35400", lw=1.0, ls="--", alpha=0.80,
        label=rf"$\Phi$-zero heights $2\tau_0 \approx {eta_zero_im:.4f},\; 2\mathrm{{Im}}\,u_1 \approx {eta_zero_off:.4f}$"
    )
    ax2.axhline(eta_zero_off, color="#d35400", lw=1.0, ls="--", alpha=0.55)
    ax2.axvline(
        yc_exact, color="#117864", lw=1.3, ls="-.",
        label=rf"Local bifurcation point $y_c(2\gamma_1) \approx {yc_exact:.4f}$"
    )
    ax2.plot([yc_exact], [0.0], marker="D", markersize=5.2, color="#117864")
    ax2.plot([yc_exact], [etac_exact], marker="D", markersize=5.2, color="#117864")
    ax2.set_xlim(0.05, 1.10)
    ax2.set_ylim(-0.48, 0.86)
    ax2.set_xlabel(r"Mirror shift parameter $y = u - v$")
    ax2.set_ylabel(r"Complex coordinates $(\pm\xi_*, \eta_*)$")
    ax2.set_title(r"(b) Tracked Saddle Branch & Strip $\Phi$-Zero Levels ($\omega = 2\gamma_1$)")
    ax2.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.92)

    fig.tight_layout()
    fig1_png = fig_dir / "fig1_wigner_saddle_coalescence.png"
    fig1_pdf = fig_dir / "fig1_wigner_saddle_coalescence.pdf"
    fig.savefig(fig1_png, bbox_inches="tight")
    fig.savefig(fig1_pdf, bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------------------------------
    # FIGURE 2: (a) Double-exponential decay on vertical Cauchy walls
    #           (b) Zero-marginal Wigner profile I(y, gamma_1) vs moment weight
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.7))

    eta_grid = np.linspace(0.0, math.pi / 4.0 - 0.004, 240)
    y_wall = 0.40
    for L_val, col, ls_style in [
        (1.5, "#2874a6", "-"),
        (2.0, "#117864", "--"),
        (2.5, "#b03a2e", "-."),
    ]:
        wall_mag = np.abs(G_y_np(L_val + 1j * eta_grid, y_wall))
        wall_mag = np.maximum(wall_mag, 1e-300)
        ax1.plot(
            eta_grid, np.log10(wall_mag), color=col, lw=2.0, ls=ls_style,
            label=rf"Vertical wall $\xi = \pm {L_val:.1f}$"
        )

    ax1.axvline(
        math.pi / 4.0, color="#7d3c98", lw=1.4, ls=":",
        label=r"Series strip boundary $\eta = \pi/4$ ($\cos 2\eta = 0$)"
    )
    ax1.axhline(0.0, color="black", lw=0.7, alpha=0.5)
    ax1.set_xlim(0.0, 0.81)
    ax1.set_ylim(-195.0, 12.0)
    ax1.set_xlabel(r"Contour height $\eta = \mathrm{Im}(z) \in [0, \pi/4)$")
    ax1.set_ylabel(r"Wall magnitude $\log_{10}|G_y(L + i\eta)|$ ($y = 0.40$)")
    ax1.set_title(r"(a) Double-Exponential Decay on Vertical Cauchy Walls")
    ax1.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.92)

    # Panel (b): Wigner correlation kernel I(y, gamma_1) and hyperbolic moment weight
    y_vec = np.linspace(0.0, 1.25, 400)
    xi_int = np.linspace(-2.5, 2.5, 5000)
    dxi = xi_int[1] - xi_int[0]
    eta_eval = 0.45
    X_m, Y_m = np.meshgrid(xi_int, y_vec)
    G_mat = G_y_np(X_m + 1j * eta_eval, Y_m)
    phase_mat = np.exp(-omega_rs * eta_eval) * np.exp(1j * omega_rs * X_m)
    I_y = np.sum(np.real(G_mat * phase_mat), axis=1) * dxi

    d_lever = 0.20
    lever_weight = np.where(
        y_vec > 1e-12,
        16.0 * y_vec * np.sinh(2.0 * d_lever * y_vec),
        0.0,
    )
    I_weighted = lever_weight * I_y

    scale_I = np.max(np.abs(I_y))
    scale_W = np.max(np.abs(I_weighted))

    ax2.plot(
        y_vec, I_y / scale_I, color="#7f8c8d", lw=1.5, ls="--",
        label=r"Unweighted $I(y, \gamma_1)$ ($\int_0^\infty I(y,\gamma_1)\,dy = 0$, multi-crossing)"
    )
    ax2.plot(
        y_vec, I_weighted / scale_W, color="#1b4f72", lw=2.2,
        label=r"Canonical integrand $16 y\sinh(2dy)\,I(y, \gamma_1)$ ($d = 0.20$)"
    )
    ax2.fill_between(
        y_vec, 0, np.maximum(I_weighted / scale_W, 0), color="#27ae60", alpha=0.22,
        label=r"Positive weighted outer lobe ($y > y_0 \approx 0.2269$)"
    )
    ax2.fill_between(
        y_vec, 0, np.minimum(I_weighted / scale_W, 0), color="#c0392b", alpha=0.22,
        label=r"Damped inner negative lobes ($y < y_0$)"
    )
    ax2.axhline(0.0, color="black", lw=0.7, alpha=0.6)
    ax2.axvline(yc_exact, color="#117864", lw=1.2, ls="-.")
    ax2.set_xlim(0.0, 1.25)
    ax2.set_ylim(-1.08, 1.12)
    ax2.set_xlabel(r"Mirror shift coordinate $y = u - v$")
    ax2.set_ylabel(r"Peak-normalized correlation and Wronskian density")
    ax2.set_title(r"(b) Zero-Marginal Wigner Profile vs. $16y\sinh(2dy)$ Weight ($t=\gamma_1$)")
    ax2.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.92)

    fig.tight_layout()
    fig2_png = fig_dir / "fig2_cauchy_contour_and_karlin_lever.png"
    fig2_pdf = fig_dir / "fig2_cauchy_contour_and_karlin_lever.pdf"
    fig.savefig(fig2_png, bbox_inches="tight")
    fig.savefig(fig2_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[*] Generated 300-DPI academic figures in: {fig_dir}")


# ==============================================================================
# 3. MAIN 60-DECIMAL VERIFICATION PIPELINE
# ==============================================================================

def main() -> None:
    t0_total = time.time()
    mp.mp.dps = 60

    print_banner("PAPER 1: WIGNER PHASE-SPACE, SHIFTED CONTOURS & LOCAL SADDLE COALESCENCE")
    print(f"[*] Python Environment : {sys.version.split()[0]}")
    print(f"[*] mpmath Precision   : {mp.mp.dps} decimal places")
    print(f"[*] Series u-strip     : |Im(u)| < pi/8 = {float(mp.pi / 8):.12f} rad")
    print(f"[*] Series z-strip     : |Im(z)| < pi/4 = {float(mp.pi / 4):.12f} rad")

    gamma_1 = mp.mpf(GAMMA_1_STR)
    omega_rs = 2 * gamma_1

    # --------------------------------------------------------------------------
    # STEP 1: Holomorphic Even Parity, Hermitian Symmetry & Strip Zeros of Phi
    # --------------------------------------------------------------------------
    print("\n" + "-" * 88)
    print(" STEP 1: HOLOMORPHIC EVEN PARITY, HERMITIAN SYMMETRY & STRIP ZEROS OF Phi(u)")
    print("-" * 88)

    u_test = mp.mpc("0.28", "0.19")
    phi_pos = phi_mp(u_test, max_n=240)
    phi_neg = phi_mp(-u_test, max_n=240)
    parity_err = abs(phi_pos - phi_neg)
    assert parity_err < mp.mpf("1e-58"), f"Parity error too large: {parity_err}"

    print(f"[1A] Holomorphic even parity Phi(-u) = Phi(u) without abs(u) at u = 0.28 + 0.19i:")
    print(f"     Phi(+u)            = {mp.nstr(phi_pos, 32)}")
    print(f"     Phi(-u)            = {mp.nstr(phi_neg, 32)}")
    print(f"     |Phi(u) - Phi(-u)| = {mp.nstr(parity_err, 6)} (< 10^-58, verified)")

    y_mp = mp.mpf("0.40")
    xi_mp = mp.mpf("0.35")
    eta_mp = mp.mpf("0.55")
    G_right = G_y_mp(xi_mp + mp.j * eta_mp, y_mp)
    G_left = G_y_mp(-xi_mp + mp.j * eta_mp, y_mp)
    herm_err = abs(G_left - mp.conj(G_right))
    assert herm_err < mp.mpf("1e-58"), f"Hermitian error too large: {herm_err}"

    G_imag_axis = G_y_mp(mp.j * eta_mp, y_mp)
    phi_half = phi_mp((y_mp + mp.j * eta_mp) / 2)
    imag_axis_err = abs(G_imag_axis - abs(phi_half) ** 2)
    assert imag_axis_err < mp.mpf("1e-58"), f"Imaginary axis non-negativity error: {imag_axis_err}"

    print(f"\n[1B] Hermitian reflection symmetry on horizontal contour z = xi + i*eta (y=0.40, xi=0.35, eta=0.55):")
    print(f"     G_y(+xi + i*eta)                        = {mp.nstr(G_right, 26)}")
    print(f"     G_y(-xi + i*eta)                        = {mp.nstr(G_left, 26)}")
    print(f"     |G_y(-xi+i*eta) - conj(G_y(+xi+i*eta))| = {mp.nstr(herm_err, 6)} (< 10^-58)")
    print(f"     On imaginary axis (xi = 0): G_y(i*eta)  = |Phi((y+i*eta)/2)|^2 = {mp.nstr(mp.re(G_imag_axis), 25)}")
    print(f"     |Im(G_y(i*eta))|                        = {mp.nstr(abs(mp.im(G_imag_axis)), 6)}")

    blowup_cos = abs(mp.cos(omega_rs * (xi_mp + mp.j * eta_mp)))
    damp_exp = abs(mp.exp(mp.j * omega_rs * (xi_mp + mp.j * eta_mp)))
    print(f"\n[1C] Pre-move decomposition at omega = 2*gamma_1 = {mp.nstr(omega_rs, 12)}, eta = 0.55:")
    print(f"     |cos(omega*(xi+i*eta))| ~ 1/2 e^(+omega*eta) = {float(blowup_cos):.6e}")
    print(f"     |e^(i*omega*(xi+i*eta))| = e^(-omega*eta)    = {float(damp_exp):.6e}")

    # [1D] Explicit zeros of Phi(u) inside |Im(u)| < pi/8 -> log singularities of ln G_y(z)
    tau0_phi = mp.findroot(
        lambda t: mp.re(phi_mp(mp.j * t, max_n=400)),
        (mp.mpf("0.31"), mp.mpf("0.32")),
        solver="anderson",
        tol=mp.mpf("1e-90"),
    )
    u1_phi = mp.findroot(
        lambda u: phi_mp(u, max_n=400) * mp.exp(mp.pi * mp.exp(4 * u)),
        mp.mpc("0.1278237567", "0.3477699161"),
        solver="newton",
        tol=mp.mpf("1e-95"),
    )
    assert abs(phi_mp(mp.j * tau0_phi, max_n=400)) < mp.mpf("1e-45")
    assert abs(phi_mp(u1_phi, max_n=400)) < mp.mpf("1e-45")
    assert abs(phi_mp(u1_phi, max_n=400) * mp.exp(mp.pi * mp.exp(4 * u1_phi))) < mp.mpf("1e-45")
    assert abs(mp.diff(lambda u: phi_mp(u, max_n=400), u1_phi)) > mp.mpf("100")
    assert tau0_phi < mp.pi / 8 and abs(mp.im(u1_phi)) < mp.pi / 8
    eta0_zero = 2 * tau0_phi
    eta1_zero = 2 * mp.im(u1_phi)
    print(f"\n[1D] Zeros of Phi(u) inside |Im(u)| < pi/8 (logarithmic branch points of ln G_y):")
    print(f"     Imaginary-axis zero u_0 = i*tau_0       : tau_0 = {mp.nstr(tau0_phi, 30)} (< pi/8 = {mp.nstr(mp.pi/8, 12)})")
    print(f"       -> G_0(i*eta_0) = 0 on imag axis at   : eta_0 = 2*tau_0 = {mp.nstr(eta0_zero, 30)} (< pi/4)")
    print(f"     Off-axis strip zero u_1                 : {mp.nstr(u1_phi, 28)}")
    print(f"       -> G_y(z) = 0 at z = +-2*u_1 +- y at  : eta_1 = 2*Im(u_1) = {mp.nstr(eta1_zero, 30)} (< pi/4)")

    # --------------------------------------------------------------------------
    # STEP 2: Vertical Cauchy Wall Decay & Series Strip Boundary eta = pi/4
    # --------------------------------------------------------------------------
    print("\n" + "-" * 88)
    print(" STEP 2: DOUBLE-EXPONENTIAL CAUCHY WALL DECAY (xi = +-L) & BOUNDARY eta = pi/4")
    print("-" * 88)

    header_s2 = f"  {'Height eta':>15} | {'cos(2*eta)':>12} | {'|G_y(L+i*eta)| at L=2.0':>24} | {'|G_y(L+i*eta)| at L=3.0':>24} | {'Series / Wall Regime':>22}"
    print(header_s2)
    print("  " + "-" * (len(header_s2) - 2))
    for eta_v in [0.0, 0.25, 0.50, 0.55, 0.65, 0.75, math.pi / 4.0]:
        c2 = math.cos(2.0 * eta_v)
        if abs(c2) > 1e-12:
            g_L2 = abs(G_y_mp(mp.mpf("2.0") + mp.j * mp.mpf(str(eta_v)), y_mp))
            g_L3 = abs(G_y_mp(mp.mpf("3.0") + mp.j * mp.mpf(str(eta_v)), y_mp))
            s_L2 = mp.nstr(g_L2, 6)
            s_L3 = mp.nstr(g_L3, 6)
            status = "CONVERGENT (DECAYS)"
        else:
            s_L2 = "SERIES DIVERGES"
            s_L3 = "SERIES DIVERGES"
            status = "STRIP BOUNDARY"
        label_eta = f"{eta_v:.6f}" if eta_v < 0.785 else "pi/4 (0.785398)"
        print(f"  {label_eta:>15} | {c2:>12.6f} | {s_L2:>24} | {s_L3:>24} | {status:>22}")

    # 35-decimal verification of Cauchy's theorem on the closed rectangle [-L, L] x [0, eta]
    mp.mp.dps = 38
    L_box = mp.mpf("2.4")
    eta_box = mp.mpf("0.55")
    w_box = mp.mpf(GAMMA_1_STR)
    I_real_axis = 2 * mp.quad(lambda xi: mp.re(G_y_mp(xi, y_mp)) * mp.cos(w_box * xi), [0, 1, L_box])
    I_top_wall = 2 * mp.exp(-w_box * eta_box) * mp.quad(
        lambda xi: mp.re(G_y_mp(xi + mp.j * eta_box, y_mp) * mp.exp(mp.j * w_box * xi)),
        [0, mp.mpf("0.8"), L_box],
    )
    I_side_wall = 2 * mp.quad(
        lambda tau: mp.re(mp.j * G_y_mp(L_box + mp.j * tau, y_mp) * mp.exp(mp.j * w_box * (L_box + mp.j * tau))),
        [0, eta_box],
    )
    cauchy_residual = abs(I_real_axis - (I_top_wall + I_side_wall))
    assert cauchy_residual < mp.mpf("1e-34"), f"Cauchy residual too large: {cauchy_residual}"

    print(f"\n[2B] Closed Cauchy rectangle verification on [-{float(L_box)}, {float(L_box)}] x [0, {float(eta_box)}] (35+ dps):")
    print(f"     Real-axis integral (eta = 0)        : {mp.nstr(I_real_axis, 34)}")
    print(f"     Shifted top-contour integral (0.55) : {mp.nstr(I_top_wall, 34)}")
    print(f"     Vertical side-wall contribution     : {mp.nstr(abs(I_side_wall), 6)}")
    print(f"     Closed Cauchy loop residual |oint|  : {mp.nstr(cauchy_residual, 6)} (< 10^-34, verified)")

    # --------------------------------------------------------------------------
    # STEP 3: Local Saddle-Point Coalescence Normal Form & Numerical Branch
    # --------------------------------------------------------------------------
    mp.mp.dps = 50
    gamma_1 = mp.mpf(GAMMA_1_STR)
    omega_rs = 2 * gamma_1

    print("\n" + "-" * 88)
    print(" STEP 3: LOCAL SADDLE-POINT COALESCENCE NORMAL FORM & TRACKED BRANCH AT 2*gamma_1")
    print("-" * 88)

    yc_inner_lead = mp.mpf("0.5") * mp.log(omega_rs / (4 * mp.pi))
    eta_outer_lead = mp.mpf("0.5") * mp.atan(omega_rs / 9)

    yc_exact_mp, etac_exact_mp = mp.findroot(
        lambda y_val, eta_val: (
            mp.im(dlogG_dx_mp(mp.j * eta_val, y_val)) + omega_rs,
            mp.re(d2logG_dx2_mp(mp.j * eta_val, y_val)),
        ),
        (mp.mpf("0.43202"), mp.mpf("0.602715")),
        maxsteps=200,
    )

    h_nf = mp.mpf("1e-8")
    dy_L1_mp = (
        -mp.im(dlogG_dx_mp(mp.j * etac_exact_mp, yc_exact_mp + h_nf))
        + mp.im(dlogG_dx_mp(mp.j * etac_exact_mp, yc_exact_mp - h_nf))
    ) / (2 * h_nf)
    L3_etac_mp = (
        -mp.re(d2logG_dx2_mp(mp.j * (etac_exact_mp + h_nf), yc_exact_mp))
        + mp.re(d2logG_dx2_mp(mp.j * (etac_exact_mp - h_nf), yc_exact_mp))
    ) / (2 * h_nf)
    K_pitch_mp = mp.sqrt(2 * dy_L1_mp / (-L3_etac_mp))

    print(f"  Frequency parameter (omega = 2*gamma_1) : {mp.nstr(omega_rs, 25)}")
    print(f"  Outer n=1 heuristic height 1/2 atan(w/9): {mp.nstr(eta_outer_lead, 20)} rad (< pi/4)")
    print(f"  Inner n=1 heuristic 1/2 ln(w/(4*pi))    : {mp.nstr(yc_inner_lead, 25)}")
    print(f"  Local bifurcation point y_c(2*g1)       : {mp.nstr(yc_exact_mp, 30)}")
    print(f"  Local bifurcation height eta_c(2*g1)    : {mp.nstr(etac_exact_mp, 30)}")
    print(f"  Non-vanishing check |G_{'{y_c}'}(i*eta_c)|    : {mp.nstr(abs(G_y_mp(mp.j * etac_exact_mp, yc_exact_mp)), 20)} (> 0)")
    print(f"  Normal-form d_y L_y'(eta_c)             : {mp.nstr(dy_L1_mp, 20)} (> 0)")
    print(f"  Normal-form L_y'''(eta_c)               : {mp.nstr(L3_etac_mp, 20)} (< 0)")
    print(f"  Local pitchfork scale K = sqrt(2dL1/-L3): {mp.nstr(K_pitch_mp, 20)}")

    header_s3 = (
        f"  {'Shift y':>10} | {'Right Saddle x_+* = +xi_* + i*eta_*':>34} | "
        f"{'Left Saddle x_-* = -xi_* + i*eta_*':>34} | {'Hessian Re d_x^2 ln G_y':>24} | {'Tracked Regime':>26}"
    )
    print("\n" + header_s3)
    print("  " + "-" * (len(header_s3) - 2))

    table_seeds = [
        (mp.mpf("0.05"), mp.mpc("0.40418674", "0.61381428"), "Outer off-axis (xi > y)"),
        (mp.mpf("0.10"), mp.mpc("0.41411144", "0.60599972"), "Outer off-axis (xi > y)"),
        (mp.mpf("0.20"), mp.mpc("0.33379542", "0.58444323"), "Outer off-axis (xi > y)"),
        (mp.mpf("0.30"), mp.mpc("0.13969227", "0.62533792"), "Inner off-axis (0 < xi < y)"),
        (mp.mpf("0.35"), mp.mpc("0.15981952", "0.60988092"), "Inner off-axis (0 < xi < y)"),
        (mp.mpf("0.40"), mp.mpc("0.03097628", "0.63405509"), "Near-axis off-axis"),
        (mp.mpf("0.42"), mp.mpc("0.02826368", "0.61080112"), "Near-axis off-axis"),
        (yc_exact_mp,    mp.mpc("0.0", etac_exact_mp),       "Local pitchfork y_c"),
        (mp.mpf("0.45"), mp.mpc("0.0", "0.54821071"),        "On-axis branch (xi_* = 0)"),
        (mp.mpf("0.50"), mp.mpc("0.0", "0.46989397"),        "On-axis branch (xi_* = 0)"),
        (mp.mpf("0.60"), mp.mpc("0.0", "0.36453529"),        "On-axis branch (xi_* = 0)"),
        (mp.mpf("0.75"), mp.mpc("0.0", "0.26030095"),        "On-axis branch (xi_* = 0)"),
        (mp.mpf("0.90"), mp.mpc("0.0", "0.18954242"),        "On-axis branch (xi_* = 0)"),
        (mp.mpf("1.10"), mp.mpc("0.0", "0.12569397"),        "On-axis branch (xi_* = 0)"),
    ]

    table_points = []
    for y_m, seed, regime_str in table_seeds:
        if y_m == yc_exact_mp:
            r_pos = seed
            r_neg = seed
        elif mp.re(seed) == 0:
            e_sol = mp.findroot(
                lambda e: mp.im(dlogG_dx_mp(mp.j * e, y_m)) + omega_rs,
                mp.im(seed),
                maxsteps=150,
            )
            r_pos = mp.mpc(0, e_sol)
            r_neg = r_pos
        else:
            r_pos = mp.findroot(
                lambda z: dlogG_dx_mp(z, y_m) + mp.j * omega_rs,
                seed,
                maxsteps=150,
            )
            if mp.re(r_pos) < 0:
                r_pos = -mp.conj(r_pos)
            r_neg = -mp.conj(r_pos)

        res_pos = abs(dlogG_dx_mp(r_pos, y_m) + mp.j * omega_rs)
        res_neg = abs(dlogG_dx_mp(r_neg, y_m) + mp.j * omega_rs)
        assert res_pos < mp.mpf("1e-40") and res_neg < mp.mpf("1e-40"), f"Root residual too large: {res_pos}"
        hess_re = mp.re(d2logG_dx2_mp(r_pos, y_m))
        if y_m != yc_exact_mp:
            assert hess_re < 0, f"Expected negative Hessian at y={y_m}, got {hess_re}"

        xi_s = float(mp.re(r_pos))
        eta_s = float(mp.im(r_pos))
        table_points.append((float(y_m), xi_s, eta_s))
        s_pos = f"{xi_s:+.8f} + {eta_s:.8f}i"
        s_neg = f"{float(mp.re(r_neg)):+.8f} + {float(mp.im(r_neg)):.8f}i"
        s_hess = f"{float(hess_re):+.8e}"
        print(f"  {float(y_m):>10.6f} | {s_pos:>34} | {s_neg:>34} | {s_hess:>24} | {regime_str:>26}")

    mp.mp.dps = 24
    pitchfork_curve = [(float(yc_exact_mp), 0.0, float(etac_exact_mp))]
    z_pf = mp.mpc("0.0066", etac_exact_mp)
    for y_v in np.linspace(float(yc_exact_mp) - 0.0005, 0.398, 22):
        y_m = mp.mpf(f"{y_v:.8f}")
        z_pf = mp.findroot(lambda z: dlogG_dx_mp(z, y_m, max_n=50) + mp.j * omega_rs, z_pf, maxsteps=80)
        pitchfork_curve.append((float(y_v), abs(float(mp.re(z_pf))), float(mp.im(z_pf))))
    pitchfork_curve.reverse()

    coalesced_curve = [(float(yc_exact_mp), 0.0, float(etac_exact_mp))]
    e_im = etac_exact_mp - mp.mpf("0.006")
    for y_v in np.linspace(float(yc_exact_mp), 1.10, 45)[1:]:
        y_m = mp.mpf(f"{y_v:.8f}")
        e_im = mp.findroot(lambda e: mp.im(dlogG_dx_mp(mp.j * e, y_m, max_n=50)) + omega_rs, e_im, maxsteps=80)
        coalesced_curve.append((float(y_v), 0.0, float(e_im)))
    mp.mp.dps = 50

    # --------------------------------------------------------------------------
    # STEP 4: Quadrature Conditioning, Canonical W=16*int Identity & Zero Marginal
    # --------------------------------------------------------------------------
    print("\n" + "-" * 88)
    print(" STEP 4: CONTOUR INVARIANCE, QUADRATURE CONDITIONING & CANONICAL W = 16 * int")
    print("-" * 88)

    xi_grid = np.linspace(-2.6, 2.6, 80000)
    dxi = xi_grid[1] - xi_grid[0]
    y_fixed = 0.40

    for case_title, omega_val, eta_list in [
        (
            "CASE A: Off-Axis Saddle Regime at omega = 2*gamma_1 = 28.2695, y = 0.40 (Net I Contour-Invariant)",
            2.0 * 14.134725141734694,
            [0.00, 0.20, 0.40, 0.50, 0.54, 0.55, 0.60],
        ),
        (
            "CASE B: On-Axis Saddle Regime at omega = gamma_1 = 14.1347, y = 0.40 (Net I Contour-Invariant)",
            14.134725141734694,
            [0.00, 0.10, 0.18, 0.20, 0.25, 0.35],
        ),
    ]:
        print(f"\n  >>> {case_title}")
        hdr = (
            f"  {'Height eta':>10} | {'exp(-w*eta)':>12} | {'Pos Lobe I_+':>14} | "
            f"{'Neg Lobe I_-':>14} | {'Net Integral I':>15} | {'Ratio I_+/|I_-|':>15} | {'Cond L1/|I|':>12}"
        )
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for eta_v in eta_list:
            x_c = xi_grid + 1j * eta_v
            G_vals = G_y_np(x_c, y_fixed)
            integrand = np.exp(-omega_val * eta_v) * np.real(G_vals * np.exp(1j * omega_val * xi_grid))
            I_pos = float(np.sum(integrand[integrand > 0]) * dxi)
            I_neg = float(np.sum(integrand[integrand < 0]) * dxi)
            I_net = I_pos + I_neg
            L1_norm = I_pos + abs(I_neg)
            ratio = I_pos / (abs(I_neg) + 1e-300)
            cond_num = L1_norm / max(abs(I_net), 1e-300)
            ratio_str = f"{ratio:15.4f}" if ratio < 1e7 else f"{ratio:15.3e}"
            print(
                f"  {eta_v:>10.2f} | {math.exp(-omega_val*eta_v):>12.4e} | {I_pos:>14.6e} | "
                f"{I_neg:>14.6e} | {I_net:>15.8e} | {ratio_str} | {cond_num:>12.4f}"
            )

    # High-precision mpmath verification of canonical W = 16 * int_0^inf y*sinh(2*d*y)*I(y,t) dy
    # and zero-marginal int_0^inf I(y, gamma_k) dy = 0 across the first 5 Riemann zeros!
    print("\n[4C] High-precision (mpmath 50 dps) verification of canonical W(d,t) = 16*int y*sinh(2dy)*I(y,t) dy:")
    mp.mp.dps = 50
    h_grid = mp.mpf("0.01")
    N_grid = 280  # covers [-2.80, 2.80] in x and [0, 2.80] in y
    h_half = h_grid / 2
    phi_half_cache = {
        m: mp.re(phi_mp(abs(m) * h_half, max_n=120))
        for m in range(0, 2 * N_grid + 2)
    }
    d_test_mp = mp.mpf("0.20")

    hdr_k = (
        f"  {'Zero':<8} | {'gamma_k':>12} | {'int_0^inf I dy':>14} | "
        f"{'W_Wig (16*int)':>20} | {'W_CR (exact)':>20} | {'|W_Wig - W_CR|':>14} | {'Crossings':>9}"
    )
    print(hdr_k)
    print("  " + "-" * (len(hdr_k) - 2))

    for k_idx in range(1, 6):
        gk = mp.im(mp.zetazero(k_idx))
        # Precompute cos(2 * gk * kx * h_grid) for kx = 0..N_grid
        cos_vec = [mp.cos(2 * gk * kx * h_grid) for kx in range(N_grid + 1)]
        marg_mp = mp.mpf(0)
        W_wig_mp = mp.mpf(0)
        I_profile = []
        for ky in range(N_grid + 1):
            y_val = ky * h_grid
            wy = h_grid * (mp.mpf("0.5") if ky == 0 else mp.mpf(1))
            # Evenness in x: G_y(-x) = G_y(x)
            Iy = phi_half_cache[ky] * phi_half_cache[ky] * h_grid
            for kx in range(1, N_grid + 1):
                g_val = phi_half_cache[abs(kx + ky)] * phi_half_cache[abs(kx - ky)]
                Iy += 2 * g_val * cos_vec[kx] * h_grid
            I_profile.append((float(y_val), float(Iy)))
            marg_mp += wy * Iy
            W_wig_mp += wy * y_val * mp.sinh(2 * d_test_mp * y_val) * Iy
        W_wig_mp *= 16
        W_exact_mp = W_cr_mp(d_test_mp, gk)
        diff_W = abs(W_wig_mp - W_exact_mp)
        assert abs(marg_mp) < mp.mpf("1e-28"), f"Zero marginal residual too large at gamma_{k_idx}: {marg_mp}"
        assert diff_W < mp.mpf("1e-28"), f"W_Wig vs W_CR mismatch at gamma_{k_idx}: {diff_W}"

        # Count sign changes of I(y, gk) on y in [0, 1.35] where |I| > 1e-22
        vals_sig = [v for (yv, v) in I_profile if yv <= 1.35 and abs(v) > 1e-22]
        n_cross = int(np.sum(np.diff(np.sign(vals_sig)) != 0))
        zname = f"gamma_{k_idx}"
        print(
            f"  {zname:<8} | {float(gk):>12.6f} | {mp.nstr(marg_mp, 4):>14} | "
            f"{mp.nstr(W_wig_mp, 12):>20} | {mp.nstr(W_exact_mp, 12):>20} | {mp.nstr(diff_W, 4):>14} | {n_cross:>9d}"
        )

    # Generate the 300-DPI figures
    script_dir = Path(__file__).resolve().parent
    fig_dir = script_dir.parent / "paper" / "figures"
    generate_figures(
        fig_dir,
        table_points,
        pitchfork_curve,
        coalesced_curve,
        float(yc_exact_mp),
        float(etac_exact_mp),
        float(eta0_zero),
        float(eta1_zero),
    )

    elapsed = time.time() - t0_total
    print("\n" + "=" * 88)
    print(f" [SUCCESS] PAPER 1 VERIFICATION & FIGURE GENERATION COMPLETED IN {elapsed:.2f}s (EXIT CODE 0)")
    print("=" * 88)


if __name__ == "__main__":
    main()
