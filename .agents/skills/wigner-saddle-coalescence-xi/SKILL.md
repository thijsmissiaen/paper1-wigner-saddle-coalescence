---
name: wigner-saddle-coalescence-xi
description: >-
  Mathematical knowledge and verification skill for Paper 1 ("Wigner Phase-Space Representation,
  Shifted Cauchy Contours, and Local Saddle-Point Coalescence in the Transverse Wronskian of
  Riemann's Xi-Function"). Activate when working with the canonical Wigner autocorrelation
  representation W(d,t) = 16 * int_0^infty y * sinh(2dy) * I(y,t) dy, shifted Cauchy contour
  quadrature conditioning in |Im(z)| < pi/4, strip zeros of Phi(u), or the local saddle-point
  coalescence normal form.
---

# Wigner Phase-Space & Shifted Cauchy Contour Skill (`wigner-saddle-coalescence-xi`)

This skill records the canonical normalizations, contour identities, conditional local saddle normal form, and explicit analytical obstructions established in Paper 1.

---

## 1. Canonical Definitions and Mirror Autocorrelation

1. **Completed Riemann $\Xi$-Function and Jacobi-Theta Kernel $\Phi(u)$:**
   With $s = \frac{1}{2} + d + \mathrm{i}t$ and $z = t - \mathrm{i}d$,

$$\Xi(t-\mathrm{i}d) = 8\int_0^\infty \Phi(u)\cos\bigl(2(t-\mathrm{i}d)u\bigr) \mathrm{d}u = 4\int _ {-\infty}^\infty \Phi(u) \mathrm{e}^{2(d + \mathrm{i}t) u} \mathrm{d}u,$$

$$\Phi(u) = \sum _ {n=1}^\infty \left(2\pi^2 n^4 \mathrm{e}^{9u} - 3\pi n^2 \mathrm{e}^{5u}\right)\exp\left(-\pi n^2 \mathrm{e}^{4u}\right).$$

2. **Canonical Wigner Scaling ($x = u+v, y = u-v$):**
   With the standard Wigner distribution $\mathcal{W} _ \Phi(q,p) := \frac{1}{2\pi}\int _ {\mathbb{R}}\Phi(q+\frac{x}{2})\Phi(q-\frac{x}{2})\mathrm{e}^{-\mathrm{i}px} \mathrm{d}x$,

$$I(y,t) := \int _ {-\infty}^\infty \Phi\left(\frac{x+y}{2}\right)\Phi\left(\frac{x-y}{2}\right)\cos(2t x) \mathrm{d}x = 2\pi \mathcal{W} _ \Phi\left(\frac{y}{2}, 2t\right),$$

$$\lvert\Xi(t-\mathrm{i}d)\rvert^2 = 16\int_0^\infty \cosh(2d y) I(y,t) \mathrm{d}y, \qquad W(d,t) = 16\int_0^\infty y\sinh(2d y) I(y,t) \mathrm{d}y,$$

$$\widetilde{W}(d,t) := \frac{W(d,t)}{d} = 32\int_0^\infty y^2 \frac{\sinh(2d y)}{2d y} I(y,t) \mathrm{d}y.$$

## 2. Shifted Contours, Local Coalescence, and Critical Pitfalls to Avoid

1. **Contour Invariance vs. Conditioning:** Shifting $z = \xi + \mathrm{i}\eta$ with $\eta \in (0,\pi/4)$ leaves the net integral $I(y,t)$ invariant by Cauchy's theorem; it improves $L^1$ quadrature conditioning ($\kappa _ {\mathrm{quad}} = 867.6 \to 1.0008$ at $y=0.40, \omega=2\gamma_1$), never the mathematical sign of $I(y,t)$.
2. **Zeros of $\Phi(u)$ inside $\lvert\mathrm{Im}(u)\rvert < \pi/8$:** $\Phi(u)$ vanishes inside the open strip, including at $u_0 = \mathrm{i}\tau_0$ ($\tau_0 \approx 0.3194150268 < \pi/8$, giving $G_0(2\mathrm{i}\tau_0) = 0$ at $\eta_0 \approx 0.638830$) and off-axis at $u_1 \approx 0.127824 + 0.347770\mathrm{i}$ ($2\mathrm{Im}(u_1) \approx 0.695540 < \pi/4$). These zeros do not affect Cauchy shifts of $G_y(z)\mathrm{e}^{2\mathrm{i}tz}$, but create logarithmic singularities in $\ln G_y(z)$ and poles in $\partial_z\ln G_y(z)$.
3. **Multi-Crossing & Davenport–Heilbronn Obstructions:** $I(y,\gamma_k)$ has $1$ sign change at $\gamma_1, \gamma_2$ but $5, 6, 7$ sign changes at $\gamma_3, \gamma_4, \gamma_5$, and the Wigner representation holds identically for Davenport–Heilbronn functions possessing off-critical zeros.

## 3. Verification Execution

```bash
python scripts/verify_paper1_wigner_saddle_coalescence.py
```
