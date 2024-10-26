# Bures-Wasserstein
This repository presents an efficient PyTorch implementation of the Bures-Wasserstein distance (the Wasserstein distance between multivariate Gaussian distributions) when the square roots of the matrices are known.

# POT implementation
The POT implementation in the version 0.9.4 (the latest at October 26, 2024) has a lot of shortcomings such as the lack of possibility to use another thing than the covarience matrix, the impossibility to use the function with a batch (a workaround can be found using torch.vmap but it is still annoying), the instability of the algorithm (see below) for not well-conditionned matrices and the inefficient calculation. 

# Our implementation
The Bures-Wasserstein distance is $bw(\Sigma,\Sigma')^2 = Tr(\Sigma + \Sigma') - 2 Tr((\Sigma^{1/2} \Sigma' \Sigma^{1/2})^{1/2})$. Hence we have two terms to compute using the square root of $\Sigma$ (resp. $\Sigma'$), denoted $A$ (resp. $B$) in the sequel.

## $Tr(A^2)$
To compute this term, we have multiple choices. The most natural one would be torch.trace(A@A) but it seems inefficient as we compute the full matrix product even though we only use the diagonal terms. The natural solution to this would be to use torch.einsum('ij,ji->', sig1, sig1). However, the implementation of torch.einsum also suffer from a lot of shortcomings due to the way matrix products are implemented in PyTorch (see https://github.com/pytorch/pytorch/issues/101249). Hence we should actually expect bad results from this optimization.

## $Tr((A B^2 A)^{1/2})$
To compute this term, we rewrite it as $Tr(((B A)^T(A B))^{1/2})$. Therefore, using torch.linalg.svdvals, we obtain the singular values $s_i$ of $B A$ and so we get $Tr((A B^2 A)^{1/2}) = \sum_i s_i$.

# Comparison on a benchmark of 10000 random matrices $3 \times 3$

| Implementation | Computation time (ms) | NaN frequency (%) |
|----------------|-----------------------|-------------------|
| POT            |       2550            |         49.91     |
| BW1            |       2425            |          0        |
| BW2            |       3295            |          0        |
| BW3            |       3008            |          0        |
| BW4            |       1129            |          0        |
