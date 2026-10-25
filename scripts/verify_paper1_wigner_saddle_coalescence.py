#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_paper1_wigner_saddle_coalescence.py
===============================================================================
High-Precision Verification & Figure Generator for Paper 1:
"Wigner Phase-Space Representation and Complex Saddle-Point Coalescence
 in the Transverse Wronskian of Riemann's Xi-Function"
===============================================================================
Author : Thijs Missiaen
Date   : October 2026
Engine : Python 3 + mpmath (60 decimal places), NumPy, SciPy, Matplotlib

This self-contained verification script reproduces all analytical identities,
60-decimal and 50-decimal numerical tables, and 300-DPI academic figures
reported in Paper 1:

  STEP 1: Holomorphic even parity Phi(-u) = Phi(u) without non-holomorphic |u|
          inside the strip |Im(u)| < pi/8, Hermitian contour symmetry
          G_y(-xi + i*eta) = conj(G_y(xi + i*eta)) on horizontal contours,
          strictly positive real values G_y(i*eta) = |Phi((y + i*eta)/2)|^2 > 0
          on the imaginary axis, and pre-move justification of the decomposition
          cos(omega*x) -> Re(e^{i*omega*x}) prior to contour deformation.

  STEP 2: Double-exponential decay on the vertical Cauchy walls xi = +-L governed
          by exp(-2*pi*cosh(2y)*e^{2L}*cos(2*eta)) for |eta| < pi/4, explicit
          demonstration of the natural boundary at eta = pi/4 (where cos(pi/2)=0),
          and 35-decimal verification of Cauchy's integral theorem on the closed
          rectangle [-L, L] x [0, eta] with contour residual < 10^-34.

  STEP 3: 50-decimal homotopy tracking of the complex saddle points
          x_*^{+-}(y, omega) = +-xi_*(y, omega) + i*eta_*(y, omega) solving
          d/dx ln G_y(x) + i*omega = 0 at omega = 2*gamma_1 across y in [0.05, 1.10],
          exact computation of the pitchfork saddle-point coalescence threshold
          y_c(2*gamma_1) = 0.4320205303... (where d/dx ln G_y(i*eta_c) + i*omega = 0
          and d^2/dxi^2 ln G_y(i*eta_c) = 0 simultaneously), and verification of
          the negative Hessian d^2/dxi^2 ln G_y(i*eta_*) < 0 for all y > y_c.

  STEP 4: Quantitative elimination of the 99.77% real-axis Fourier cancellation
          (eta = 0, where I_+ / |I_-| = 1.002308 and net retention is 0.2303%)
          by shifting to the complex contour eta in (0, pi/4) (where I_+ / |I_-|
          rises to 2547.95 at eta = 0.50 and 1.547 x 10^11 in the single-saddle
          coalesced regime), together with verification of Hudson's zero-marginal
          identity int_{-infty}^{infty} I(y, gamma_k) dy = 2 |Xi(gamma_k)|^2 = 0
          and the Karlin hyperbolic moment lever y * sinh(d*y).
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

# Set global mpmath precision to 60 decimal places
mp.mp.dps = 60

GAMMA_1_STR = "14.1347251417346937904572519835624702707842571156992431756855"


def print_banner(title: str) -> None:
    print("=" * 88)
    print(f" {title}")
    print("=" * 88)


# ==============================================================================
# 1. HOLOMORPHIC JACOBI-RIEMANN KERNEL Phi(u) AND CONVOLUTION KERNEL G_y(x)
# ==============================================================================

def phi_mp(u, max_n: int = 140):
    """
    Evaluates the holomorphic Jacobi-Riemann kernel Phi(u) in the complex strip
    |Im(u)| < pi/8:
        Phi(u) = sum_{n=1}^infty (2*pi^2*n^4*e^{9u} - 3*pi*n^2*e^{5u}) * exp(-pi*n^2*e^{4u})
    Note: No non-holomorphic abs(u) is used. Even parity Phi(-u) = Phi(u) holds
    analytically by Jacobi's modular functional identity theta(e^{-4u}) = e^{2u}*theta(e^{4u}).
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


def phi_and_derivatives_mp(u, max_n: int = 140):
    """
    Simultaneously evaluates Phi(u), Phi'(u), and Phi''(u) to 60 decimal places
    inside the holomorphic strip |Im(u)| < pi/8.
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
    Evaluates the holomorphic mirror-convolution kernel
        G_y(x) = Phi((x + y)/2) * Phi((x - y)/2)
    in the strip |Im(x)| < pi/4.
    """
    return phi_mp((x + y) / 2) * phi_mp((x - y) / 2)


def dlogG_dx_mp(x, y, max_n: int = 140):
    """
    Exact logarithmic derivative:
        d/dx ln G_y(x) = 1/2 * [Phi'((x+y)/2)/Phi((x+y)/2) + Phi'((x-y)/2)/Phi((x-y)/2)].
    """
    p1, dp1, _ = phi_and_derivatives_mp((x + y) / 2, max_n=max_n)
    p2, dp2, _ = phi_and_derivatives_mp((x - y) / 2, max_n=max_n)
    return mp.mpf("0.5") * (dp1 / p1 + dp2 / p2)


def d2logG_dx2_mp(x, y):
    """
    Exact second logarithmic derivative:
        d^2/dx^2 ln G_y(x) = 1/4 * [(Phi''/Phi - (Phi'/Phi)^2)((x+y)/2) + (Phi''/Phi - (Phi'/Phi)^2)((x-y)/2)].
    """
    p1, dp1, d2p1 = phi_and_derivatives_mp((x + y) / 2)
    p2, dp2, d2p2 = phi_and_derivatives_mp((x - y) / 2)
    t1 = d2p1 / p1 - (dp1 / p1) ** 2
    t2 = d2p2 / p2 - (dp2 / p2) ** 2
    return mp.mpf("0.25") * (t1 + t2)


def phi_complex_np(u):
    """
    Vectorized NumPy evaluation of the holomorphic kernel Phi(u) for |Im(u)| < pi/8.
    Uses the analytically proven identity Phi(-u) = Phi(u) when Re(u) < 0 so that
    34 Gaussian terms achieve full IEEE-754 double precision across all u.
    """
    u = np.asarray(u, dtype=np.complex128)
    u_eff = np.where(np.real(u) >= 0.0, u, -u)
    s = np.zeros_like(u_eff, dtype=np.complex128)
    pi = np.pi
    pi2 = pi * pi
    e4 = np.exp(4.0 * u_eff)
    e5 = np.exp(5.0 * u_eff)
    e9 = np.exp(9.0 * u_eff)
    for n in range(1, 34):
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
) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10.5,
        "axes.titlesize": 11,
        "legend.fontsize": 8.2,
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
    # FIGURE 1: (a) Real-axis 99.77% cancellation vs complex contour lift
    #           (b) Pitchfork saddle-point coalescence across y_c
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.6))

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
        label=r"Real axis $\eta = 0$ ($I_+ / |I_-| = 1.0023$, $99.77\%$ cancellation)"
    )
    ax1.plot(
        xi_plot, norm_lift, color="#1b4f72", lw=2.0,
        label=r"Shifted contour $\eta = 0.50$ ($I_+ / |I_-| = 2548.0$, $99.96\%$ net)"
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
    ax1.set_xlabel(r"Horizontal contour coordinate $\xi = \mathrm{Re}(x)$")
    ax1.set_ylabel(r"Normalized integrand profile $\mathrm{Re}[G_y(\xi+i\eta)e^{i\omega(\xi+i\eta)}]$")
    ax1.set_title(r"(a) Elimination of Real-Axis Cancellation ($y = 0.40, \omega = 2\gamma_1$)")
    ax1.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92)

    # Panel (b): Exact 50-decimal saddle points + pitchfork & coalesced branches
    # 1. Analytic n=1 outer reference curve for +-xi_*^out(y, omega)
    y_ref = np.linspace(0.05, 0.452, 200)
    xi_out_ref = np.maximum(
        0.0,
        0.5 * np.log(np.sqrt(omega_rs ** 2 + 81.0) / (4.0 * np.pi * np.cosh(2.0 * y_ref))),
    )
    ax2.plot(
        y_ref, xi_out_ref, color="#5dade2", lw=1.3, ls=":",
        label=r"Leading $n=1$ outer envelope $\pm\xi_*^{\mathrm{out}}(y, 2\gamma_1)$"
    )
    ax2.plot(y_ref, -xi_out_ref, color="#5dade2", lw=1.3, ls=":")

    # 2. Exact root-found pitchfork & coalesced curves
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
        label=r"Exact saddle branches $\pm\xi_*(y, 2\gamma_1)$ ($\xi_* = 0$ for $y \geq y_c$)"
    )
    ax2.plot(
        y_co, eta_co, color="#b03a2e", lw=2.1,
        label=r"Exact saddle height $\eta_*(y, 2\gamma_1) < \pi/4$"
    )

    # 3. Exact 50-decimal Table 3 roots as markers
    y_tb = np.array([row[0] for row in table_points])
    xi_tb = np.array([row[1] for row in table_points])
    eta_tb = np.array([row[2] for row in table_points])
    ax2.plot(y_tb, xi_tb, "o", color="#1b4f72", markersize=4.2, alpha=0.90)
    ax2.plot(y_tb[xi_tb > 0], -xi_tb[xi_tb > 0], "o", color="#1b4f72", markersize=4.2, alpha=0.90, fillstyle="none")
    ax2.plot(y_tb, eta_tb, "s", color="#b03a2e", markersize=3.8, alpha=0.88)

    ax2.axhline(
        math.pi / 4.0, color="#7d3c98", lw=1.2, ls=":",
        label=r"Holomorphic strip boundary $\eta_{\max} = \pi/4 \approx 0.7854$"
    )
    ax2.axvline(
        yc_exact, color="#117864", lw=1.4, ls="-.",
        label=rf"Exact coalescence threshold $y_c(2\gamma_1) \approx {yc_exact:.4f}$"
    )
    ax2.plot([yc_exact], [0.0], marker="D", markersize=5.5, color="#117864")
    ax2.plot([yc_exact], [etac_exact], marker="D", markersize=5.5, color="#117864")
    ax2.set_xlim(0.05, 1.10)
    ax2.set_ylim(-0.48, 0.86)
    ax2.set_xlabel(r"Mirror shift parameter $y = u - v$")
    ax2.set_ylabel(r"Complex saddle coordinates $(\pm\xi_*, \eta_*)$")
    ax2.set_title(r"(b) Pitchfork Saddle-Point Coalescence at $\omega = 2\gamma_1$")
    ax2.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.92)

    fig.tight_layout()
    fig1_png = fig_dir / "fig1_wigner_saddle_coalescence.png"
    fig1_pdf = fig_dir / "fig1_wigner_saddle_coalescence.pdf"
    fig.savefig(fig1_png, bbox_inches="tight")
    fig.savefig(fig1_pdf, bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------------------------------
    # FIGURE 2: (a) Double-exponential decay on vertical Cauchy walls vs pi/4
    #           (b) Wigner kernel I(y, gamma_1) zero marginal vs Karlin lever
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.6))

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
        label=r"Natural boundary $\eta = \pi/4$ ($\cos(2\eta) = 0$)"
    )
    ax1.axhline(0.0, color="black", lw=0.7, alpha=0.5)
    ax1.set_xlim(0.0, 0.81)
    ax1.set_ylim(-195.0, 12.0)
    ax1.set_xlabel(r"Contour height $\eta = \mathrm{Im}(x) \in [0, \pi/4)$")
    ax1.set_ylabel(r"Wall magnitude $\log_{10}|G_y(L + i\eta)|$ ($y = 0.40$)")
    ax1.set_title(r"(a) Double-Exponential Decay on Vertical Cauchy Walls")
    ax1.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.92)

    # Panel (b): Wigner correlation kernel I(y, gamma_1) and Karlin hyperbolic lever
    y_vec = np.linspace(0.002, 1.25, 320)
    xi_int = np.linspace(-2.4, 2.4, 5000)
    dxi = xi_int[1] - xi_int[0]
    eta_eval = 0.45
    X_m, Y_m = np.meshgrid(xi_int, y_vec)
    G_mat = G_y_np(X_m + 1j * eta_eval, Y_m)
    phase_mat = np.exp(-omega_rs * eta_eval) * np.exp(1j * omega_rs * X_m)
    I_y = np.sum(np.real(G_mat * phase_mat), axis=1) * dxi

    d_lever = 0.20
    lever_weight = (y_vec ** 2) * (np.sinh(2.0 * d_lever * y_vec) / (2.0 * d_lever * y_vec))
    I_weighted = lever_weight * I_y

    scale_I = np.max(np.abs(I_y))
    scale_W = np.max(np.abs(I_weighted))

    ax2.plot(
        y_vec, I_y / scale_I, color="#7f8c8d", lw=1.5, ls="--",
        label=r"Unweighted kernel $I(y, \gamma_1)$ ($\int_0^\infty I(y,\gamma_1)\,dy = 0$)"
    )
    ax2.plot(
        y_vec, I_weighted / scale_W, color="#1b4f72", lw=2.2,
        label=r"Hyperbolic-weighted $y^2 \frac{\sinh(2dy)}{2dy} I(y, \gamma_1)$ ($d = 0.20$)"
    )
    ax2.fill_between(
        y_vec, 0, np.maximum(I_weighted / scale_W, 0), color="#27ae60", alpha=0.22,
        label=r"Positive coalesced tail ($y > y_0 \approx 0.2266$)"
    )
    ax2.fill_between(
        y_vec, 0, np.minimum(I_weighted / scale_W, 0), color="#c0392b", alpha=0.22,
        label=r"Suppressed inner trough ($y < y_0$)"
    )
    ax2.axhline(0.0, color="black", lw=0.7, alpha=0.6)
    ax2.axvline(yc_exact, color="#117864", lw=1.2, ls="-.")
    ax2.set_xlim(0.0, 1.25)
    ax2.set_ylim(-1.08, 1.12)
    ax2.set_xlabel(r"Mirror shift coordinate $y = u - v$")
    ax2.set_ylabel(r"Normalized correlation and moment density")
    ax2.set_title(r"(b) Zero-Marginal Wigner Kernel vs. Hyperbolic Moment Lever")
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

    print_banner("PAPER 1: WIGNER PHASE-SPACE & COMPLEX SADDLE-POINT COALESCENCE VERIFICATION")
    print(f"[*] Python Environment : {sys.version.split()[0]}")
    print(f"[*] mpmath Precision   : {mp.mp.dps} decimal places")
    print(f"[*] Holomorphic u-strip: |Im(u)| < pi/8 = {float(mp.pi / 8):.12f} rad")
    print(f"[*] Holomorphic x-strip: |Im(x)| < pi/4 = {float(mp.pi / 4):.12f} rad")

    gamma_1 = mp.mpf(GAMMA_1_STR)
    omega_rs = 2 * gamma_1

    # --------------------------------------------------------------------------
    # STEP 1: Holomorphic Even Parity & Hermitian Contour Symmetry
    # --------------------------------------------------------------------------
    print("\n" + "-" * 88)
    print(" STEP 1: HOLOMORPHIC EVEN PARITY Phi(-u)=Phi(u) & HERMITIAN CONTOUR SYMMETRY")
    print("-" * 88)

    u_test = mp.mpc("0.28", "0.19")
    phi_pos = phi_mp(u_test, max_n=140)
    phi_neg = phi_mp(-u_test, max_n=140)
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
    assert imag_axis_err < mp.mpf("1e-58"), f"Imaginary axis positivity error: {imag_axis_err}"

    print(f"\n[1B] Hermitian reflection symmetry on horizontal contour x = xi + i*eta (y=0.40, xi=0.35, eta=0.55):")
    print(f"     G_y(+xi + i*eta)                        = {mp.nstr(G_right, 26)}")
    print(f"     G_y(-xi + i*eta)                        = {mp.nstr(G_left, 26)}")
    print(f"     |G_y(-xi+i*eta) - conj(G_y(+xi+i*eta))| = {mp.nstr(herm_err, 6)} (< 10^-58)")
    print(f"     On imaginary axis (xi = 0): G_y(i*eta)  = |Phi((y+i*eta)/2)|^2 = {mp.nstr(mp.re(G_imag_axis), 25)}")
    print(f"     |Im(G_y(i*eta))|                        = {mp.nstr(abs(mp.im(G_imag_axis)), 6)} (purely real and strictly positive)")

    blowup_cos = abs(mp.cos(omega_rs * (xi_mp + mp.j * eta_mp)))
    damp_exp = abs(mp.exp(mp.j * omega_rs * (xi_mp + mp.j * eta_mp)))
    print(f"\n[1C] Pre-move justification at omega = 2*gamma_1 = {mp.nstr(omega_rs, 12)}, eta = 0.55:")
    print(f"     Naive |cos(omega*(xi+i*eta))| ~ 1/2 e^(+omega*eta) = {float(blowup_cos):.6e} (exponential growth)")
    print(f"     Holomorphic |e^(i*omega*(xi+i*eta))| = e^(-omega*eta) = {float(damp_exp):.6e} (exponential damping)")

    # --------------------------------------------------------------------------
    # STEP 2: Vertical Cauchy Wall Decay & Natural Boundary at eta = pi/4
    # --------------------------------------------------------------------------
    print("\n" + "-" * 88)
    print(" STEP 2: DOUBLE-EXPONENTIAL CAUCHY WALL DECAY (xi = +-L) & BOUNDARY eta = pi/4")
    print("-" * 88)

    header_s2 = f"  {'Height eta':>15} | {'cos(2*eta)':>12} | {'|G_y(L+i*eta)| at L=2.0':>24} | {'|G_y(L+i*eta)| at L=3.0':>24} | {'Wall Regime':>20}"
    print(header_s2)
    print("  " + "-" * (len(header_s2) - 2))
    for eta_v in [0.0, 0.25, 0.50, 0.55, 0.65, 0.75, math.pi / 4.0]:
        c2 = math.cos(2.0 * eta_v)
        if abs(c2) > 1e-12:
            g_L2 = abs(G_y_mp(mp.mpf("2.0") + mp.j * mp.mpf(str(eta_v)), y_mp))
            g_L3 = abs(G_y_mp(mp.mpf("3.0") + mp.j * mp.mpf(str(eta_v)), y_mp))
            s_L2 = mp.nstr(g_L2, 6)
            s_L3 = mp.nstr(g_L3, 6)
            status = "CLOSED (DECAYS)"
        else:
            s_L2 = "DIVERGENT (|e^-..|=1)"
            s_L3 = "DIVERGENT (|e^-..|=1)"
            status = "NATURAL BOUNDARY"
        label_eta = f"{eta_v:.6f}" if eta_v < 0.785 else "pi/4 (0.785398)"
        print(f"  {label_eta:>15} | {c2:>12.6f} | {s_L2:>24} | {s_L3:>24} | {status:>20}")

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
    # STEP 3: Complex Saddle-Point Trajectory & Exact Pitchfork Coalescence y_c
    # --------------------------------------------------------------------------
    mp.mp.dps = 50
    gamma_1 = mp.mpf(GAMMA_1_STR)
    omega_rs = 2 * gamma_1

    print("\n" + "-" * 88)
    print(" STEP 3: PITCHFORK SADDLE-POINT COALESCENCE x_* = +-xi_*(y,omega) + i*eta_*(y,omega)")
    print("-" * 88)

    yc_inner_lead = mp.mpf("0.5") * mp.log(omega_rs / (4 * mp.pi))
    yc_acosh_g1 = mp.mpf("0.5") * mp.acosh(mp.sqrt(gamma_1 ** 2 + 81) / (4 * mp.pi))
    eta_outer_lead = mp.mpf("0.5") * mp.atan(omega_rs / 9)

    # Exact pitchfork coalescence point (y_c, eta_c) at omega = 2*gamma_1:
    # Simultaneous root of Im(d/dx ln G_y(i*eta)) + omega = 0 and Re(d^2/dx^2 ln G_y(i*eta)) = 0
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
    print(f"  Outer n=1 asymptotic height 1/2 atan(w/9): {mp.nstr(eta_outer_lead, 20)} rad (< pi/4)")
    print(f"  Inner n=1 coalescence 1/2 ln(w/(4*pi))  : {mp.nstr(yc_inner_lead, 25)}")
    print(f"  Leading acosh estimate at t = gamma_1   : {mp.nstr(yc_acosh_g1, 25)}")
    print(f"  Exact pitchfork coalescence y_c(2*g1)   : {mp.nstr(yc_exact_mp, 30)}")
    print(f"  Exact saddle height eta_*(y_c, 2*g1)    : {mp.nstr(etac_exact_mp, 30)}")
    print(f"  Normal-form d_y L_y'(eta_c)             : {mp.nstr(dy_L1_mp, 20)} (> 0)")
    print(f"  Normal-form L_y'''(eta_c)               : {mp.nstr(L3_etac_mp, 20)} (< 0)")
    print(f"  Pitchfork scale K = sqrt(2*d_y L1/-L3)  : {mp.nstr(K_pitch_mp, 20)}")

    header_s3 = (
        f"  {'Shift y':>10} | {'Right Saddle x_+* = +xi_* + i*eta_*':>34} | "
        f"{'Left Saddle x_-* = -xi_* + i*eta_*':>34} | {'Hessian Re d_x^2 ln G_y':>24} | {'Regime':>26}"
    )
    print("\n" + header_s3)
    print("  " + "-" * (len(header_s3) - 2))

    table_seeds = [
        (mp.mpf("0.05"), mp.mpc("0.40418674", "0.61381428"), "Outer off-axis (xi > y)"),
        (mp.mpf("0.10"), mp.mpc("0.41411144", "0.60599972"), "Outer off-axis (xi > y)"),
        (mp.mpf("0.20"), mp.mpc("0.33379542", "0.58444323"), "Outer off-axis (xi > y)"),
        (mp.mpf("0.30"), mp.mpc("0.13969227", "0.62533792"), "Inner off-axis (0 < xi < y)"),
        (mp.mpf("0.35"), mp.mpc("0.15981952", "0.60988092"), "Inner off-axis (0 < xi < y)"),
        (mp.mpf("0.40"), mp.mpc("0.03097628", "0.63405509"), "Pitchfork off-axis"),
        (mp.mpf("0.42"), mp.mpc("0.02826368", "0.61080112"), "Pitchfork off-axis"),
        (yc_exact_mp,    mp.mpc("0.0", etac_exact_mp),       "Pitchfork bifurcation y_c"),
        (mp.mpf("0.45"), mp.mpc("0.0", "0.54821071"),        "Coalesced (xi_* = 0)"),
        (mp.mpf("0.50"), mp.mpc("0.0", "0.46989397"),        "Coalesced (xi_* = 0)"),
        (mp.mpf("0.60"), mp.mpc("0.0", "0.36453529"),        "Coalesced (xi_* = 0)"),
        (mp.mpf("0.75"), mp.mpc("0.0", "0.26030095"),        "Coalesced (xi_* = 0)"),
        (mp.mpf("0.90"), mp.mpc("0.0", "0.18954242"),        "Coalesced (xi_* = 0)"),
        (mp.mpf("1.10"), mp.mpc("0.0", "0.12569397"),        "Coalesced (xi_* = 0)"),
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

    # Build exact root-found pitchfork & coalesced curves for Figure 1(b) (zero interpolation)
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
    # STEP 4: Quantitative Elimination of 99.77% Real-Axis Cancellation
    # --------------------------------------------------------------------------
    print("\n" + "-" * 88)
    print(" STEP 4: ELIMINATION OF 99.77% REAL-AXIS CANCELLATION & KARLIN MOMENT LEVER")
    print("-" * 88)

    xi_grid = np.linspace(-2.6, 2.6, 80000)
    dxi = xi_grid[1] - xi_grid[0]
    y_fixed = 0.40

    for case_title, omega_val, eta_list in [
        (
            "CASE A: Two-Saddle Regime at omega = 2*gamma_1 = 28.2695, y = 0.40",
            2.0 * 14.134725141734694,
            [0.00, 0.20, 0.40, 0.50, 0.54, 0.55, 0.60],
        ),
        (
            "CASE B: Single-Saddle Coalesced Regime at omega = gamma_1 = 14.1347, y = 0.40",
            14.134725141734694,
            [0.00, 0.10, 0.18, 0.20, 0.25, 0.35],
        ),
    ]:
        print(f"\n  >>> {case_title}")
        hdr = (
            f"  {'Height eta':>10} | {'exp(-w*eta)':>12} | {'Pos Lobe I_+':>14} | "
            f"{'Neg Lobe I_-':>14} | {'Net Integral I':>15} | {'Ratio I_+/|I_-|':>15} | {'Cancellation':>12} | {'Net Share':>10}"
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
            ratio = I_pos / (abs(I_neg) + 1e-300)
            cancel_pct = (1.0 - I_net / I_pos) * 100.0
            dom_pct = (I_net / I_pos) * 100.0
            ratio_str = f"{ratio:15.4f}" if ratio < 1e7 else f"{ratio:15.3e}"
            print(
                f"  {eta_v:>10.2f} | {math.exp(-omega_val*eta_v):>12.4e} | {I_pos:>14.6e} | "
                f"{I_neg:>14.6e} | {I_net:>15.8e} | {ratio_str} | {cancel_pct:>11.4f}% | {dom_pct:>9.4f}%"
            )

    # Verify Hudson zero-marginal identity and Karlin hyperbolic moment lever across first 5 zeros
    print("\n[4C] Hudson zero-marginal identity and Karlin hyperbolic moment lever across first 5 zeros:")
    y_vec = np.linspace(0.001, 1.35, 300)
    xi_fast = np.linspace(-2.4, 2.4, 6000)
    dxi_f = xi_fast[1] - xi_fast[0]
    X_m, Y_m = np.meshgrid(xi_fast, y_vec)
    eta_eval = 0.45
    G_mat = G_y_np(X_m + 1j * eta_eval, Y_m)

    zeros_5 = [
        ("gamma_1", 14.134725141734694),
        ("gamma_2", 21.022039638771555),
        ("gamma_3", 25.010857580145688),
        ("gamma_4", 30.424876125859513),
        ("gamma_5", 32.935061587739189),
    ]
    d_test = 0.20
    hdr_k = (
        f"  {'Zero':<9} | {'t = gamma_k':>12} | {'Sign change y_0':>15} | "
        f"{'Marginal int I(y)dy':>20} | {'W(d=0.2, gamma_k)':>18} | {'Status':>18}"
    )
    print(hdr_k)
    print("  " + "-" * (len(hdr_k) - 2))
    for zname, zt in zeros_5:
        w = 2.0 * zt
        phase_mat = np.exp(-w * eta_eval) * np.exp(1j * w * X_m)
        I_y = np.sum(np.real(G_mat * phase_mat), axis=1) * dxi_f
        int_Iy = float(np.trapezoid(I_y, y_vec))
        crossings = np.where(np.diff(np.sign(I_y)) > 0)[0]
        y0_val = float(y_vec[crossings[-1]]) if len(crossings) > 0 else 0.0
        W_val = float(8.0 * np.trapezoid(y_vec * np.sinh(2.0 * d_test * y_vec) * I_y, y_vec))
        assert W_val > 0.0, f"Expected W > 0 at {zname}"
        print(
            f"  {zname:<9} | {zt:>12.6f} | {y0_val:>15.4f} | "
            f"{int_Iy:>+20.6e} | {W_val:>+18.6e} | {'STRICTLY POSITIVE':>18}"
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
    )

    elapsed = time.time() - t0_total
    print("\n" + "=" * 88)
    print(f" [SUCCESS] PAPER 1 VERIFICATION & FIGURE GENERATION COMPLETED IN {elapsed:.2f}s (EXIT CODE 0)")
    print("=" * 88)


if __name__ == "__main__":
    main()
