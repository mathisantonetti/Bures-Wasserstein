# Bures-Wasserstein
This repository presents an efficient PyTorch implementation of the Bures-Wasserstein distance (the Wasserstein distance between multivariate Gaussian distributions) when the square roots of the matrices are known.

# POT implementation
The POT implementation in the version 0.9.4 (the latest at October 26, 2024) has a lot of shortcomings such as the lack of possibility to use another thing than the covariance matrix, the impossibility to use the function with a batch (a workaround can be found using torch.vmap but it is still annoying), the instability of the algorithm (see below) for not well-conditioned matrices and the inefficient calculation. 

# Our implementation
The Bures-Wasserstein distance is $bw(\Sigma,\Sigma')^2 = Tr(\Sigma + \Sigma') - 2 Tr((\Sigma^{1/2} \Sigma' \Sigma^{1/2})^{1/2})$. Hence we have two terms to compute using the square root of $\Sigma$ (resp. $\Sigma'$), denoted $A$ (resp. $B$) in the sequel.

## $Tr(A^2)$
To compute this term, we have multiple choices. The most natural one would be torch.trace(A@A) but it seems inefficient as we compute the full matrix product even though we only use the diagonal terms. The natural solution to this would be to use torch.einsum('ij,ji->', sig1, sig1). However, the implementation of torch.einsum also suffers from a lot of shortcomings due to the way matrix products are implemented in PyTorch (see https://github.com/pytorch/pytorch/issues/101249). Hence we should actually expect bad results from this optimization.

## $Tr((A B^2 A)^{1/2})$
To compute this term, we rewrite it as $Tr(((B A)^T(A B))^{1/2})$. The most natural way to do this would be the computation of the square root but it seems inefficient to do this only to sum up the diagonal terms. Therefore, using torch.linalg.svdvals, we obtain the singular values $s_i$ of $B A$ and so we get $Tr((A B^2 A)^{1/2}) = \sum_i s_i$.

# Comparison on a benchmark of 10000 random matrices $3 \times 3$

To test the implementation, we use a set of 10000 randomly generated well-conditioned $3 \times 3$ matrices ($\kappa = 10$) for the time computation and not so well-conditioned matrices ($\kappa = 10^4$ vs $\kappa = 10$) for the stability (NaN frequency). 

| Implementation | Computation time (ms) | NaN frequency (%) |
|----------------|-----------------------|-------------------|
| POT            |       2550            |         49.91     |
| BW1            |       2425            |          0        |
| BW2            |       3295            |          0        |
| BW3            |       3008            |          0        |
| BW4            |       1129            |          0        |

We can see that the POT function gives a NaN $\approx 50$% of the time because of the implementation instability. If we retry with well-conditioned matrices, the NaN frequency drops to $0 \%$. The implementation BW1 using the natural implementations improves efficiency a bit and is more stable. We can see that the implementation BW2 that uses torch.einsum is inefficient due to the PyTorch implementation. Indeed, if we change torch.einsum(t) by torch.tensor(np.einsum(t.detach().cpu().numpy())) (which is the difference between BW2 and BW3), we obtain a better computation time even though changing the tensor to a numpy array to a tensor is inefficient. We see that the implementation BW4 is $\approx 2.25$ times faster than the POT implementation on a CPU. The result may be different on a GPU.

We can see by changing the condition number that it is responsible for the NaN returned by the POT function.
|   Condition Number  | $\kappa = 10$ | $\kappa = 10^2$ | $\kappa = 10^3$ | $\kappa = 10^4$ | $\kappa = 10^5$ | $\kappa = 10^6$ |
|-------------------|---------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| NaN frequency (%) |       0       |         0       |         29.35    |         50.28    |         50.21    |         53.98    |

# Comparison on a benchmark of random matrices of different dimensions
The following table shows the  influence of the dimension on the computation efficiency difference between the different implementations. The data is the computation time in ms.
| Implementation | $3 \times 3$ | $30 \times 30$ | $300 \times 300$ | $3000 \times 3000$ |
|----------------|--------------|----------------|------------------|--------------------|
| POT            |       2550   |    4709        | 996              | 5539               |
| BW1            |       2425   |          4621  | 29225            | 15123              |
| BW2            |       3295   |          5458  | 27054            | 16428              |
| BW3            |       3008   |          5099  | 26815            | 18285              |
| BW4            |       1129   |          1627  | 781              | 2070               |

We can see that the implementations that use the homemade square root function (i.e. BW1, BW2, BW3) are considerably slow compared to the others which was to be expected since the square root function in the code has not been optimized enough to compete with the scipy built in function. We see that a weird thing happens in middle/high-dimension (between $30 \times 30$ and $3000 \times 3000$) that makes the implementations POT and BW4 faster for those dimensions specifically. This is probably due to the way matrix products are optimized since this optimization is for machine/deep learning that generally use matrices of this size.

## A last attempt
To make up for the absence of a reasonable torch.einsum function, we can try to implement manually a pseudo einsum. The resulting implementation is denoted BW5.
| Implementation | $30 \times 30$ | $300 \times 300$ | $3000 \times 3000$ |
|----------------|--------------|----------------|------------------|
| POT            |      4760        | 10761              | 24420               |
| BW4            |       1600  | 6941              | 11378               |
| BW5            |       1589  | 6972              | 10137               |

We see that our implementation outperforms a bit BW4 on very high dimensions but is comparable on lower dimensions (which should not be the case with a fully optimized einsum). We still get an interesting alternative to torch.trace(A@A).
