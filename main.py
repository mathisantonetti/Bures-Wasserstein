import torch
import numpy as np
from unit_tests import test_generatePSDMatrix, test_bures_wasserstein_stability, test_bures_wasserstein_coherence, test_bures_wasserstein_time


# First table
torch.manual_seed(0)
np.random.seed(0)
test_generatePSDMatrix(10000, cond_P=10**4)
test_generatePSDMatrix(10000, cond_P=10**1)
test_bures_wasserstein_stability(10000, try_bures_wasserstein=True)
test_bures_wasserstein_coherence(10000)
test_bures_wasserstein_time(10000, only_usefuls=False, with_grad=True)

# Second table (condition influence)
torch.manual_seed(0)
np.random.seed(0)
test_bures_wasserstein_stability(10000, cond_P2=10**1)
test_bures_wasserstein_stability(10000, cond_P2=10**2)
test_bures_wasserstein_stability(10000, cond_P2=10**3)
test_bures_wasserstein_stability(10000, cond_P2=10**4)
test_bures_wasserstein_stability(10000, cond_P2=10**5)
test_bures_wasserstein_stability(10000, cond_P2=10**6)

# Comparison on different dimensions
test_generatePSDMatrix(10000, n=30, cond_P=10**4)
test_bures_wasserstein_time(10000, n=30, only_usefuls=False, with_grad=True)
test_bures_wasserstein_time(100, n=300, only_usefuls=False, with_grad=True)
test_bures_wasserstein_time(1, n=3000, only_usefuls=False, with_grad=True)
