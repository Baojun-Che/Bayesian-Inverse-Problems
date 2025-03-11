# Bayesian-Inverse-Problems
Codes for Bayesian inverse problems.
Some of codes are from https://github.com/PKU-CMEGroup/InverseProblems.jl

Under constuction!

* [DF_GMVI](Inversion/DF_GMVI.jl) (source code), [DG-GMVI](Derivative-Free-Variational-Inference/DerivativeFreeVariationalInference.ipynb)(tutorial): see [link](https://arxiv.org/abs/2501.04259) for the paper.

* [Affine Invariant MCMC](Inversion/AffineInvariantMCMC.jl)
* [BBVI-1](Inversion/BBVI.jl) uses the idea of BBVI to simulate Gaussian mixture nature gradient flow. And differently, [BBVI-2](Inversion/BBVI-inv.jl) simulate the the gradient flow of $C^{-1}$.
* The folder [LowRankGaussian](LowRankGaussian) explores using a rank-scalar form $\varepsilon I+QQ^\top$ to approximate either $C$ (approximate in each step in [CovApprox-1](LowRankGaussian/LowRankBBVI.jl), and derive a new gradient flow in [CovApprox-2](LowRankGaussian/LowRankBBVI-2.jl)) or $C^{-1}$ (in [CovInvApprox-1](LowRankGaussian/RSI-BBVI.jl)).
