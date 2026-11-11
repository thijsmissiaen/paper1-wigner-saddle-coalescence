# Wigner Phase-Space Representation, Shifted Cauchy Contours, and Local Saddle-Point Coalescence in the Transverse Wronskian of Riemann's $\Xi$-Function

**DOI (Zenodo / CERN):** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23637324.svg)](https://doi.org/10.5281/zenodo.23637324) [`10.5281/zenodo.23637324`](https://doi.org/10.5281/zenodo.23637324)  
**Author:** Thijs Missiaen (Independent Researcher, Belgium) &middot; **ORCID:** [`0009-0007-3678-0810`](https://orcid.org/0009-0007-3678-0810)  
**Repository:** [`https://github.com/thijsmissiaen/paper1-wigner-saddle-coalescence`](https://github.com/thijsmissiaen/paper1-wigner-saddle-coalescence)  
**Manuscript:** [`paper/paper1_wigner_saddle_coalescence.pdf`](./paper/paper1_wigner_saddle_coalescence.pdf) (10 pages; AMS-LaTeX source: [`paper/paper1_wigner_saddle_coalescence.tex`](./paper/paper1_wigner_saddle_coalescence.tex))  
**Verification Script:** [`scripts/verify_paper1_wigner_saddle_coalescence.py`](./scripts/verify_paper1_wigner_saddle_coalescence.py) (60-decimal `mpmath` + 300-DPI figure generator)

---

> [!NOTE]
> **Scope and Analytical Distinction:** This note separates (i) exact Wigner autocorrelation and Cauchy contour identities for $W(d,t) = \frac{1}{2}\partial_d\lvert\Xi(t-\mathrm{i}d)\rvert^2$, (ii) the conditional local normal form for complex stationary-point coalescence away from zeros of $\Phi(u)$, and (iii) explicit obstructions inside $\lvert\mathrm{Im}(z)\rvert < \pi/4$ (strip zeros of $\Phi(u)$ creating logarithmic branch points in $\ln G_y(z)$, multi-crossing oscillations of $I(y,\gamma_k)$ for $k \ge 3$, and Davenport–Heilbronn counterexamples). Shifting a horizontal contour preserves the net integral $I(y,t)$ identically by Cauchy's theorem while improving $L^1$ quadrature conditioning.

## Abstract and Mathematical Summary

We study the phase-space and complex-contour structure of the transverse Wronskian $W(d,t) = \frac{1}{2}\partial_d\lvert\Xi(t-\mathrm{i}d)\rvert^2$ of Riemann's completed zeta function $\Xi(t-\mathrm{i}d) = \xi(\frac{1}{2}+d+\mathrm{i}t)$.

1. **Canonical Wigner Representation and Zero-Marginal Identity:**  
   In mirror coordinates $x = u + v$ and $y = u - v$, the transverse Wronskian and its parity-reduced counterpart $\widetilde{W}(d,t) = W(d,t)/d$ satisfy the exact canonical identities

$$W(d,t) = 16\int_0^\infty y\sinh(2d y) I(y,t) \mathrm{d}y, \qquad \widetilde{W}(d,t) = 32\int_0^\infty y^2 \frac{\sinh(2d y)}{2d y} I(y,t) \mathrm{d}y,$$

   where $I(y,t)$ is proportional to the standard quantum-mechanical Wigner distribution $\mathcal{W} _ \Phi(q,p) = \frac{1}{2\pi}\int _ {\mathbb{R}}\Phi(q+\frac{x}{2})\Phi(q-\frac{x}{2})\mathrm{e}^{-\mathrm{i}px} \mathrm{d}x$ evaluated at $(q,p) = (y/2, 2t)$:

$$I(y,t) := \int _ {-\infty}^\infty \Phi\left(\frac{x+y}{2}\right)\Phi\left(\frac{x-y}{2}\right)\cos(2t x) \mathrm{d}x = 2\pi \mathcal{W} _ \Phi\left(\frac{y}{2}, 2t\right).$$

   At every real zero $\gamma_k$ of $\Xi$, $\int_0^\infty I(y,\gamma_k) \mathrm{d}y = \frac{1}{16}\lvert\Xi(\gamma_k)\rvert^2 = 0$, and Hudson's theorem (1974) implies that $I(y,t)$ changes sign in $\mathbb{R}^2$.

2. **Holomorphic Strip $\lvert\mathrm{Im}(z)\rvert < \pi/4$, Contour Conditioning, and Strip Zeros of $\Phi(u)$:**  
   Using $\Phi(u) = \frac{1}{16}(\partial_u^2 - 1)[\mathrm{e}^u\theta(\mathrm{e}^{4u})]$ and $\theta(\mathrm{e}^{-4u}) = \mathrm{e}^{2u}\theta(\mathrm{e}^{4u})$, $\Phi(-u) = \Phi(u)$ holds holomorphically on $\lvert\mathrm{Im}(u)\rvert < \pi/8$. On vertical walls $\xi = \pm L$, $G_y(z) = \Phi(\frac{z+y}{2})\Phi(\frac{z-y}{2})$ decays double-exponentially for $\lvert\eta\rvert < \pi/4$, whereas the defining series diverges on $\lvert\eta\rvert = \pi/4$ ($\cos 2\eta = 0$). Shifting the horizontal contour reduces the $L^1$ quadrature condition number $\kappa _ {\mathrm{quad}}$ at $(y,\omega) = (0.40, 2\gamma_1)$ from $867.6$ ($\eta = 0$) to $1.0008$ ($\eta = 0.50$) while keeping $I(0.40,\gamma_1)$ invariant. Crucially, $\Phi(u)$ has zeros inside $\lvert\mathrm{Im}(u)\rvert < \pi/8$ (including $u_0 = \mathrm{i}\tau_0$ with $\tau_0 \approx 0.3194150268$ and $u_1 \approx 0.1278237567 + 0.3477699161\mathrm{i}$), creating logarithmic branch points in $\ln G_y(z)$ inside $\mathcal{S} _ {\pi/4}$.

3. **Local Saddle-Point Coalescence Normal Form and Explicit Obstructions:**  
   Away from zeros of $\Phi(u)$, stationary points of $\ln G_y(z) + \mathrm{i}\omega z$ colliding at $(y_c, \mathrm{i}\eta_c)$ obey the local Weierstrass quadratic law $(z _ \star^\pm - \mathrm{i}\eta_c)^2 = K^2(y_c - y) + \mathcal{O}(\lvert y - y_c\rvert^{3/2})$ (rotating $90^\circ$ from horizontal off-axis pairs for $y < y_c$ to imaginary-axis roots for $y > y_c$; at $\omega = 2\gamma_1$, $y_c \approx 0.4320205303$, $\eta_c \approx 0.6027150239$, $K \approx 0.2991933645$). For $k \ge 3$, $I(y,\gamma_k)$ exhibits multiple sign changes ($5$ crossings at $\gamma_3$), and the entire Wigner representation applies unchanged to Davenport–Heilbronn kernels with off-critical zeros.

---

## Reproducing the Verification and Figures

```bash
python scripts/verify_paper1_wigner_saddle_coalescence.py
```

| Verification Step | Mathematical Quantity | Verified Value / Residual | Precision |
| :--- | :--- | :--- | :--- |
| **Step 1A–B** | Holomorphic parity $\lvert\Phi(u) - \Phi(-u)\rvert$ & Hermitian symmetry | $2.61 \times 10^{-61}$, $1.17 \times 10^{-59}$ | $60$ dps |
| **Step 1D** | Strip zeros of $\Phi(u)$ ($\tau_0 < \pi/8$, $\mathrm{Im}(u_1) < \pi/8$) | $\tau_0 = 0.3194150268\dots$, $2\mathrm{Im}(u_1) = 0.6955398322\dots$ | $60$ dps |
| **Step 2B** | Closed Cauchy rectangle residual $\lvert\oint G_y(z)\mathrm{e}^{\mathrm{i}\gamma_1 z} \mathrm{d}z\rvert$ | $3.59 \times 10^{-43}$ | $38$ dps |
| **Step 3** | Local coalescence point $(y_c(2\gamma_1), \eta_c(2\gamma_1))$ | $y_c = 0.4320205303\dots$, $\eta_c = 0.6027150239\dots$ | $50$ dps |
| **Step 4A** | Quadrature condition number $\kappa _ {\mathrm{quad}}$ ($y=0.40, \eta = 0 \to 0.50$) | $867.6007 \to 1.0008$ (net $I = 1.76903968 \times 10^{-6}$) | IEEE-754 |
| **Step 4C** | Canonical $W _ {\mathrm{Wig}}(0.20,\gamma_k) = 16\int y\sinh(2dy)I \mathrm{d}y$ vs. $W _ {\mathrm{CR}}$ | Max error $2.15 \times 10^{-54}$; $\lvert\int_0^\infty I \mathrm{d}y\rvert \le 1.54 \times 10^{-53}$ | $50$ dps |

## Citation

Please cite this work using the metadata in [`CITATION.cff`](./CITATION.cff).
