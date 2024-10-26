import numpy as np
import torch
import time
from ot.gaussian import bures_wasserstein_distance
from distance import bures_wasserstein_v1, bures_wasserstein_v2, bures_wasserstein_v3, bures_wasserstein_v4

def generatePSDMatrix(n = 3, cond_P = 10**4):
    # Piece of code taken from Bartomoleo Stellato : random_mat_condition_number.py
    log_cond_P = np.log(cond_P)
    exp_vec = np.arange(-log_cond_P/4., log_cond_P * (n + 1)/(4 * (n - 1)), log_cond_P/(2.*(n-1)))
    s = np.exp(exp_vec)
    S = np.diag(s)
    U, _ = np.linalg.qr((np.random.rand(n, n) - 5.) * 200)
    V, _ = np.linalg.qr((np.random.rand(n, n) - 5.) * 200)
    P = U.dot(S).dot(V.T)
    P = P.dot(P.T)
    return torch.tensor(P/np.max(np.abs(P)), dtype=torch.float32)

def test_generatePSDMatrix(numtests, n=3, cond_P=10**9):
    quotient = []
    for i in range(numtests):
        Mat = generatePSDMatrix(n = n, cond_P=cond_P)
        vals = torch.linalg.svdvals(Mat)
        #print(vals, Mat.shape)
        quotient.append(((torch.max(vals)/torch.min(vals))/cond_P).item())
    
    print("min quotient real_cond/desired_cond : ", np.min(quotient), " | max quotient real_cond/desired_cond : ", np.max(quotient))

def test_bures_wasserstein_stability(numtests, n=3, cond_P1=10**1, cond_P2=10**4):
    numnans = [0, 0, 0, 0, 0]
    for i in range(numtests):
        mu1, mu2 = torch.randn(n), torch.randn(n)
        sig1_12, sig2_12 = generatePSDMatrix(n=n, cond_P=cond_P1), generatePSDMatrix(n=n, cond_P=cond_P2)
        numnans[0] += torch.isnan(bures_wasserstein_distance(mu1, mu2, sig1_12@sig1_12, sig2_12@sig2_12)).item()
        numnans[1] += torch.isnan(bures_wasserstein_v1(mu1, mu2, sig1_12, sig2_12)).item()
        numnans[2] += torch.isnan(bures_wasserstein_v2(mu1, mu2, sig1_12, sig2_12)).item()
        numnans[3] += torch.isnan(bures_wasserstein_v3(mu1, mu2, sig1_12, sig2_12)).item()
        numnans[4] += torch.isnan(bures_wasserstein_v4(mu1, mu2, sig1_12, sig2_12)).item()
    
    print("pot : ", numnans[0], " nans over ", numtests, " tests...")
    print("bw1 : ", numnans[1], " nans over ", numtests, " tests...")
    print("bw2 : ", numnans[2], " nans over ", numtests, " tests...")
    print("bw3 : ", numnans[3], " nans over ", numtests, " tests...")
    print("bw4 : ", numnans[4], " nans over ", numtests, " tests...")

def test_bures_wasserstein_coherence(numtests, n=3, cond_P1=10**1, cond_P2=10**1):
    mean, std = [], []
    for i in range(numtests):
        mu1, mu2 = torch.randn(n), torch.randn(n)
        sig1_12, sig2_12 = generatePSDMatrix(n=n, cond_P=cond_P1), generatePSDMatrix(n=n, cond_P=cond_P2)
        pot_res = (bures_wasserstein_distance(mu1, mu2, sig1_12@sig1_12, sig2_12@sig2_12)**2).item()
        bw1_res = bures_wasserstein_v1(mu1, mu2, sig1_12, sig2_12).item()
        bw2_res = bures_wasserstein_v2(mu1, mu2, sig1_12, sig2_12).item()
        bw3_res = bures_wasserstein_v3(mu1, mu2, sig1_12, sig2_12).item()
        bw4_res = bures_wasserstein_v4(mu1, mu2, sig1_12, sig2_12).item()

        mean.append((pot_res+bw1_res+bw2_res+bw3_res+bw4_res)/5)
        std.append(np.sqrt((pot_res**2+bw1_res**2+bw2_res**2+bw3_res**2+bw4_res**2)/5 - mean[-1]**2)/mean[-1])
    
    print("max std : ", np.max(std), " | median std : ", np.median(std))

def test_bures_wasserstein_time(numtests, n=3, cond_P1=10**1, cond_P2=10**1, only_usefuls=False):
    print("Only usefuls selected : ", only_usefuls)
    sig1_12, sig2_12 = [generatePSDMatrix(n=n, cond_P=cond_P1) for i in range(numtests)], [generatePSDMatrix(n=n, cond_P=cond_P2) for i in range(numtests)]
    mu1, mu2 = torch.randn((numtests,n)), torch.randn((numtests,n))

    tpot = time.time()
    for i in range(numtests):
        res = bures_wasserstein_distance(mu1[i], mu2[i], sig1_12[i]@sig1_12[i], sig2_12[i]@sig2_12[i])**2

    print("duration pot : ", time.time()-tpot)

    if(not only_usefuls):
        tbw1 = time.time()
        for i in range(numtests):
            res = bures_wasserstein_v1(mu1[i], mu2[i], sig1_12[i], sig2_12[i])

        print("duration bw1 : ", time.time()-tbw1)

        tbw2 = time.time()
        for i in range(numtests):
            res = bures_wasserstein_v2(mu1[i], mu2[i], sig1_12[i], sig2_12[i])

        print("duration bw2 : ", time.time()-tbw2)

        tbw3 = time.time()
        for i in range(numtests):
            res = bures_wasserstein_v3(mu1[i], mu2[i], sig1_12[i], sig2_12[i])

        print("duration bw3 : ", time.time()-tbw3)

    tbw4 = time.time()
    for i in range(numtests):
        res = bures_wasserstein_v4(mu1[i], mu2[i], sig1_12[i], sig2_12[i])

    print("duration bw4 : ", time.time()-tbw4)

    tbw5 = time.time()
    for i in range(numtests):
        res = bures_wasserstein_v4(mu1[i], mu2[i], sig1_12[i], sig2_12[i])

    print("duration bw5 : ", time.time()-tbw5)