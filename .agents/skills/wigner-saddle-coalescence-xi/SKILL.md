---
name: wigner-saddle-coalescence-xi
description: >-
  Mathematical knowledge and verification skill for Paper 1 ("Wigner Phase-Space Representation
  and Complex Saddle-Point Coalescence in the Transverse Wronskian of Riemann's Xi-Function").
  Activate this skill when analyzing, extending, or numerically verifying the mirror-coordinate
  Wigner identity I(y,t) = 2*pi*W_Phi(y,2t), the Cauchy contour box in |Im(x)| < pi/4, Hermitian
  contour symmetry, or the pitchfork saddle-point coalescence at y_c(omega).
---

# Wigner Phase-Space & Complex Saddle-Point Coalescence Skill (`wigner-saddle-coalescence-xi`)

This skill provides AI research agents with the exact analytical identities, complex contour theorems, and 60-decimal `mpmath` verification procedures established in Paper 1.

---

## 1. Canonical Definitions and Mirror Factorization

1. **Completed Riemann $\Xi$-Function and Jacobi-Theta Kernel $\Phi(u)$:**
   With $s = \frac{1}{2} + d + \mathrm{i}t$ and $z = t - \mathrm{i}d$,

   $$\Xi(t-\mathrm{i}d) = \int_{-\infty}^\infty \Phi(u)\,\mathrm{e}^{d u - \mathrm{i}t u}\,\mathrm{d}u, \qquad \Phi(u) = \sum_{n=1}^\infty \left(2\pi^2 n^4 \mathrm{e}^{9u} - 3\pi n^2 \mathrm{e}^{5u}\right)\exp\left(-\pi n^2 \mathrm{e}^{4u}\right).$$

2. **Mirror Coordinates $(x = u+v, y = u-v)$ and Wigner Identity:**

   $$W(d,t) := \frac{1}{2}\partial_d\lvert\Xi(t-\mathrm{i}d)\rvert^2 = \frac{1}{2}\int_0^\infty y\sinh(d y)\,I(y,t)\,\mathrm{d}y, \qquad \widetilde{W}(d,t) := \frac{W(d,t)}{d} = \frac{1}{2}\int_0^\infty y^2\,\frac{\sinh(d y)}{d y}\,I(y,t)\,\mathrm{d}y,$$

   where the inner correlation integral is proportional to the quantum-mechanical Wigner distribution of $\Phi$:

   $$I(y,t) := \int_{-\infty}^\infty G_y(x)\cos(t x)\,\mathrm{d}x = 2\pi\,\mathcal{W}_\Phi(y, 2t), \qquad G_y(x) := \Phi\left(\frac{x+y}{2}\right)\Phi\left(\frac{x-y}{2}\right).$$

3. **Hudson Zero-Marginal Identity and Karlin Moment Lever:**
   At every real zero $\gamma_k$ of $\Xi$, $\int_{-\infty}^\infty I(y,\gamma_k)\,\mathrm{d}y = 2\lvert\Xi(\gamma_k)\rvert^2 = 0$. For any sign-change point $y_0 > 0$,

   $$W(d,t) = \frac{1}{2}y_0\sinh(d y_0)\lvert\Xi(t)\rvert^2 + \frac{1}{2}\int_0^\infty \big(y\sinh(d y) - y_0\sinh(d y_0)\big)I(y,t)\,\mathrm{d}y.$$

## 2. Complex Contour Rules in $\lvert\mathrm{Im}(x)\rvert < \pi/4$ and Pitchfork Coalescence

1. **Holomorphic Even Parity without `abs(u)`:**
   Inside $\lvert\mathrm{Im}(u)\rvert < \pi/8$, $\Phi(u) = \frac{1}{16}(\partial_u^2 - 1)[\mathrm{e}^u\theta(\mathrm{e}^{4u})]$ satisfies $\Phi(-u) = \Phi(u)$ identically via $\theta(\mathrm{e}^{-4u}) = \mathrm{e}^{2u}\theta(\mathrm{e}^{4u})$. Never use non-holomorphic `abs(u)` when evaluating $\Phi(u)$ in the complex plane.
2. **Pre-Move Replacement and Vertical Wall Decay:**
   Always replace $\cos(\omega x)$ by $\mathrm{Re}(\mathrm{e}^{\mathrm{i}\omega x})$ on the real axis $\eta = 0$ before shifting to $x = \xi + \mathrm{i}\eta$ ($\eta \in (0,\pi/4)$). On the vertical Cauchy walls $\xi = \pm L$, $\lvert G_y(\pm L + \mathrm{i}\tau)\rvert \le 16\pi^4 \mathrm{e}^{9L}\exp(-2\pi\cosh(2y)\mathrm{e}^{2L}\cos(2\eta)) \to 0$. Never place a contour on the natural boundary $\lvert\eta\rvert = \pi/4$ where $\cos(2\eta) = 0$.
3. **Hermitian Contour Symmetry and Pitchfork Coalescence:**
   On any horizontal contour $x = \xi + \mathrm{i}\eta$ ($\lvert\eta\rvert < \pi/4$), $G_y(-\xi+\mathrm{i}\eta) = \overline{G_y(\xi+\mathrm{i}\eta)}$ and $G_y(\mathrm{i}\eta) = \lvert\Phi(\frac{y+\mathrm{i}\eta}{2})\rvert^2 \ge 0$. At the critical threshold $y_c(\omega)$ (where $\mathrm{Im}[\partial_x\ln G_y(\mathrm{i}\eta_c)] + \omega = 0$ and $\partial_\xi^2\ln G_y(\mathrm{i}\eta_c) = 0$; at $\omega = 2\gamma_1$, $y_c \approx 0.4320205303$), the off-axis saddle pair $x_*^\pm = \pm\xi_* + \mathrm{i}\eta_*$ coalesces onto $\xi_* = 0$ with $\partial_\xi^2\ln G_y(\mathrm{i}\eta_*) < 0$ for all $\lvert y\rvert \ge y_c(\omega)$.

## 3. Verification Execution

Run the standalone 60-decimal verification script from the repository root:

```bash
python scripts/verify_paper1_wigner_saddle_coalescence.py
```
