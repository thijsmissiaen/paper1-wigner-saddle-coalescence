# Wigner Phase-Space Representation and Complex Saddle-Point Coalescence in the Transverse Wronskian of Riemann's $\Xi$-Function

**Author:** Thijs Missiaen (Independent Researcher, Belgium)  
**Repository:** `https://github.com/thijsmissiaen/paper1-wigner-saddle-coalescence`  
**Manuscript:** [`paper/paper1_wigner_saddle_coalescence.pdf`](./paper/paper1_wigner_saddle_coalescence.pdf) (AMS-LaTeX source: [`paper/paper1_wigner_saddle_coalescence.tex`](./paper/paper1_wigner_saddle_coalescence.tex))  
**Verification Script:** [`scripts/verify_paper1_wigner_saddle_coalescence.py`](./scripts/verify_paper1_wigner_saddle_coalescence.py) (60-decimal `mpmath` + 300-DPI figure generator)

---

## Abstract and Mathematical Overview

This repository contains the complete AMS-LaTeX manuscript, compiled academic PDF, high-resolution figures, and 60-decimal arbitrary-precision verification suite for Paper 1 of our study on the analytical and phase-space structure of Riemann's Jacobi-theta kernel $\Phi(u)$.

We prove that the inner correlation integral $I(y,t)$ governing the transverse Wronskian $W(d,t) = \frac{1}{2}\partial_d|\Xi(t-\mathrm{i}d)|^2$ of Riemann's completed $\Xi$-function is proportional to the quantum-mechanical Wigner phase-space distribution $\mathcal{W}_\Phi(y, 2t)$ of the kernel $\Phi(u)$, and that its $99.77\%$ real-axis Fourier cancellation at $t = \gamma_1$ is eliminated by shifting the integration contour into the holomorphic strip $|\operatorname{Im}(x)| < \pi/4$.

### Main Analytical Results (Theorem 1.1 & Theorem 4.2)

1. **Wigner--Hudson Phase-Space Representation and Zero-Marginal Identity:**  
   In mirror coordinates $x = u + v$ and $y = u - v$, the transverse Wronskian and its even parity-reduced counterpart $\widetilde{W}(d,t) = W(d,t)/d$ take the exact form
   $$W(d,t) = \frac{1}{2}\int_0^\infty y\sinh(d y)\,I(y,t)\,\mathrm{d}y, \qquad \widetilde{W}(d,t) = \frac{1}{2}\int_0^\infty y^2\,\frac{\sinh(d y)}{d y}\,I(y,t)\,\mathrm{d}y,$$
   where
   $$I(y,t) = \int_{-\infty}^\infty \Phi\!\left(\frac{x+y}{2}\right)\Phi\!\left(\frac{x-y}{2}\right)\cos(t x)\,\mathrm{d}x = 2\pi\,\mathcal{W}_\Phi(y, 2t).$$
   At every real zero $\gamma_k$ of $\Xi$, the marginal integral vanishes identically ($\int_{-\infty}^\infty I(y,\gamma_k)\,\mathrm{d}y = 2|\Xi(\gamma_k)|^2 = 0$), and Hudson's theorem (1974) implies that $I(y,t)$ must change sign in the $(y,t)$-plane because $\Phi(u)$ is non-Gaussian.

2. **Holomorphic Strip $|\operatorname{Im}(x)| < \pi/4$, Vertical Wall Decay, and Hermitian Symmetry:**  
   Using the Jacobi differential identity $\Phi(u) = \frac{1}{16}(\partial_u^2 - 1)[\mathrm{e}^u\theta(\mathrm{e}^{4u})]$ and the modular transformation $\theta(\mathrm{e}^{-4u}) = \mathrm{e}^{2u}\theta(\mathrm{e}^{4u})$, we prove $\Phi(-u) = \Phi(u)$ holomorphically inside $|\operatorname{Im}(u)| < \pi/8$ without non-holomorphic absolute values. On the vertical Cauchy walls $\xi = \pm L$, the convolution kernel $G_y(x) = \Phi(\frac{x+y}{2})\Phi(\frac{x-y}{2})$ decays double-exponentially as $\mathcal{O}(\mathrm{e}^{9L}\exp(-2\pi\cosh(2y)\mathrm{e}^{2L}\cos(2\eta)))$ for all $|\eta| < \pi/4$, whereas $|\eta| = \pi/4$ is a natural boundary where $\cos(2\eta) = 0$. Along any horizontal contour $x = \xi + \mathrm{i}\eta$, $G_y$ satisfies the exact Hermitian reflection symmetry
   $$G_y(-\xi + \mathrm{i}\eta) = \overline{G_y(\xi + \mathrm{i}\eta)}, \qquad G_y(\mathrm{i}\eta) = \left|\Phi\!\left(\frac{y+\mathrm{i}\eta}{2}\right)\right|^2 \ge 0.$$

3. **Pitchfork Saddle-Point Coalescence and Elimination of Real-Axis Cancellation:**  
   The complex stationary-phase points $x_*^\pm(y,\omega) = \pm\xi_*(y,\omega) + \mathrm{i}\eta_*(y,\omega)$ solving $\partial_x\ln G_y(x) + \mathrm{i}\omega = 0$ undergo a pitchfork bifurcation at $y_c(\omega)$ (at $\omega_1 = 2\gamma_1$: $y_c(2\gamma_1) \approx 0.4320205303$, $\eta_*(y_c,2\gamma_1) \approx 0.6027150239$). For $|y| \ge y_c(\omega)$, the conjugate saddle pair coalesces onto the imaginary axis $\xi_* = 0$ with strictly negative horizontal Hessian $\partial_\xi^2\ln G_y(\mathrm{i}\eta_*) < 0$. Shifting the contour from $\eta = 0$ to $\eta \in (0,\pi/4)$ raises the positive-to-negative lobe ratio $I_+ / |I_-|$ at $y = 0.40$ from $1.0023$ ($99.77\%$ cancellation) to $2547.95$ at $\omega = 2\gamma_1, \eta = 0.50$, and to $1.547 \times 10^{11}$ in the single-saddle coalesced regime ($\omega = \gamma_1, \eta = 0.18$).

## Repository Structure

```text
paper1-wigner-saddle-coalescence/
├── README.md
├── CITATION.cff
├── LICENSE
├── AGENT_SKILL_MANIFEST.json
├── .agents/
│   └── skills/
│       └── wigner-saddle-coalescence-xi/
│           └── SKILL.md
├── paper/
│   ├── paper1_wigner_saddle_coalescence.tex
│   ├── paper1_wigner_saddle_coalescence.pdf
│   └── figures/
│       ├── fig1_wigner_saddle_coalescence.png
│       ├── fig1_wigner_saddle_coalescence.pdf
│       ├── fig2_cauchy_contour_and_karlin_lever.png
│       └── fig2_cauchy_contour_and_karlin_lever.pdf
└── scripts/
    └── verify_paper1_wigner_saddle_coalescence.py
```

## Reproducing the 60-Decimal Verification and Figures

To run the complete 60-decimal `mpmath` verification suite and regenerate both 300-DPI figures (`PNG` and vector `PDF`):

```bash
python scripts/verify_paper1_wigner_saddle_coalescence.py
```

### Benchmark Summary Verified by `verify_paper1_wigner_saddle_coalescence.py`

| Verification Step | Mathematical Quantity | Verified Value / Residual | Precision |
| :--- | :--- | :--- | :--- |
| **Step 1A** | Holomorphic parity $\|\Phi(u) - \Phi(-u)\|$ at $u = 0.28 + 0.19\mathrm{i}$ | $2.60591 \times 10^{-61}$ | $60$ dps |
| **Step 1B** | Hermitian symmetry $\|G_y(-\xi+\mathrm{i}\eta) - \overline{G_y(\xi+\mathrm{i}\eta)}\|$ | $1.16682 \times 10^{-59}$ | $60$ dps |
| **Step 2A** | Vertical Cauchy wall magnitude $\|G_y(3.0 + 0.55\mathrm{i})\|$ ($y = 0.40$) | $2.98538 \times 10^{-654}$ | $60$ dps |
| **Step 2B** | Closed Cauchy rectangle residual $\|\oint_{\partial\mathcal{R}} G_y(z)\mathrm{e}^{\mathrm{i}\gamma_1 z}\,\mathrm{d}z\|$ | $3.58732 \times 10^{-43}$ | $38$ dps |
| **Step 3** | Exact pitchfork coalescence threshold $y_c(2\gamma_1)$ and height $\eta_c$ | $y_c = 0.4320205303\dots, \eta_c = 0.6027150239\dots$ | $50$ dps |
| **Step 4A** | Lobe dominance ratio $I_+ / \|I_-\|$ ($\omega = 2\gamma_1, y = 0.40, \eta = 0 \to 0.50$) | $1.0023 \to 2547.9547$ ($99.96\%$ net) | IEEE-754 |
| **Step 4B** | Lobe dominance ratio $I_+ / \|I_-\|$ ($\omega = \gamma_1, y = 0.40, \eta = 0 \to 0.18$) | $1.8349 \to 1.547 \times 10^{11}$ ($100.00\%$ net) | IEEE-754 |

## Citation

If you use the analytical identities, contour bounds, or verification scripts from this repository in your research, please cite this work using the metadata in [`CITATION.cff`](./CITATION.cff).
