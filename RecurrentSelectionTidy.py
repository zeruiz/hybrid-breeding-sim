import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import scipy.io
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import copy
import random


def blmmp(A, D, Y):
    r = ro.r
    r.source('/Users/zeeray/Desktop/demo.R')
    if D is not None:
        with localconverter(ro.default_converter + pandas2ri.converter):
            A_r = ro.conversion.py2rpy(A)
            D_r = ro.conversion.py2rpy(D)
            Y_r = ro.conversion.py2rpy(Y)
        BLMMP_r = ro.globalenv['BLMMP']
        df_result_r = BLMMP_r(A_r, D_r, Y_r)
        with localconverter(ro.default_converter + pandas2ri.converter):
            pd_from_r_df = ro.conversion.rpy2py(df_result_r)
        aHat = np.array(pd_from_r_df.rx('aHat')).squeeze()
        dHat = np.array(pd_from_r_df.rx('dHat')).squeeze()
        GEBVHat = np.array(pd_from_r_df.rx('GEBVHat')).squeeze()
        s2Hat = np.array(pd_from_r_df.rx('sigma2Hat')).squeeze()
        H2Hat = np.array(pd_from_r_df.rx('H2Hat')).squeeze()
        # rmse = np.sqrt(mean_squared_error(Y.loc[:, 0], yHat)) / np.mean(Y.loc[:, 0])
        return GEBVHat, aHat, dHat, s2Hat, H2Hat
    if D is None:
        with localconverter(ro.default_converter + pandas2ri.converter):
            A_r = ro.conversion.py2rpy(A)
            Y_r = ro.conversion.py2rpy(Y)
        BLMMP_r = ro.globalenv['BLMMP']
        df_result_r = BLMMP_r(A_r, 0, Y_r)
        with localconverter(ro.default_converter + pandas2ri.converter):
            pd_from_r_df = ro.conversion.rpy2py(df_result_r)
        aHat = np.array(pd_from_r_df.rx('aHat')).squeeze()
        GEBVHat = np.array(pd_from_r_df.rx('GEBVHat')).squeeze()
        s2Hat = np.array(pd_from_r_df.rx('sigma2Hat')).squeeze()
        H2Hat = np.array(pd_from_r_df.rx('H2Hat')).squeeze()
        # rrmse = np.sqrt(mean_squared_error(Y.loc[:, 0], yHat)) / np.mean(Y.loc[:, 0])
        # h2 = np.var(Y.loc[:, 0] ) / np.var(y2)
        return GEBVHat, aHat, s2Hat, H2Hat


def cross(chromo1, chromo2, k, m):
    k_mat = np.tile(np.concatenate(([0.5], k)), (2 * m, 1))
    rc_mat = np.random.random([2 * m, len(k) + 1])
    t_mat = np.sign(rc_mat - k_mat)
    g1_mat = np.tile(np.matrix([chromo1[0, :], chromo2[0, :]]), (m, 1))
    g2_mat = np.tile(np.matrix([chromo1[1, :], chromo2[1, :]]), (m, 1))
    t_pois_mat = np.cumprod(t_mat, axis=1) < 0
    result = np.where(t_pois_mat, g2_mat, g1_mat)
    result_format = result.reshape(m, 2, len(k) + 1)
    return result_format


def dFlat(a, dp, dn, pidx, nidx):
    dComplete = np.zeros(len(a))
    dComplete[pidx] = dp
    dComplete[nidx] = dn
    return dComplete


def simulator(G, a, d, H2):
    # H2 is None or 1 can both lead to GEBV=Pheno
    A = pd.DataFrame((np.sum(G, axis=1)))
    D = pd.DataFrame((np.sum(G, axis=1) == 1) * 1)
    Av = np.sum(A * a, axis=1).to_numpy()
    Dv = 0 if d.ndim == 0 else np.sum(D * d, axis=1).to_numpy()
    GEBV = Av + Dv
    if H2 is None:
        Pheno = GEBV
        sigma2 = None
    else:
        sigma2 = np.var(Av + Dv) / H2 - np.var(Av + Dv)
        Pheno = GEBV + np.random.normal(0, np.sqrt(sigma2), G.shape[0])
    return Pheno, GEBV


def reproduce(BP, RF, nps):
    nc = len(nps)
    G = np.tile(BP[0, :, :], (np.sum(nps), 1, 1))
    m = 0
    for c in range(nc):
        if nps[c] >= 1:
            G[[i + m for i in range(nps[c])], :, :] = cross(BP[2 * c, :, :], BP[2 * c + 1, :, :], RF, nps[c])
            m += nps[c]
    return G


def testcross(G1, G2, RF, S):
    # G1[1, :, :] x G2[1, :, :], G1[1, :, :] x G2[2, :, :], G1[1, :, :] x G2[3, :, :], ...
    I = G1.shape[0]
    J = G2.shape[0]
    idx = np.array(np.meshgrid([i for i in range(I)], [I + j for j in range(J)])).T.reshape(-1, 2)
    idx = idx.reshape(idx.shape[0] * idx.shape[1])
    Gall = np.concatenate([G1, G2], -0)
    G = reproduce(Gall[idx, :, :], RF, np.ones(int(len(idx) / 2), dtype=int) * S)
    return G


def pair_cross(G1, G2, RF, S):
    if G1.shape[0] == G2.shape[0]:
        n = G1.shape[0]
        progeny = np.zeros([n*S, 2, G1.shape[2]])
        for i in range(n):
            progeny[i*S:(i+1)*S, :, :] = cross(G1[i, :, :], G2[i, :, :], RF, S)
        return progeny
    else:
        return False


def gca(P, K1, K2, K):
    Vmat = np.mean(np.array_split(P, K1 * K2), axis=1).reshape(K1, K2)
    gca_value = Vmat.mean(axis=1)
    IDX = np.sort(gca_value.argsort()[::-1][:K])
    return IDX


def sca(P, K1, K2, K):
    Vmat = np.mean(np.array_split(P, K1 * K2), axis=1).reshape(K1, K2)
    gca_value = Vmat.max(axis=1)
    IDX = np.sort(gca_value.argsort()[::-1][:K])
    return IDX


def sca_new(P, K1, K2, K, evalTarget):
    Vmat = np.mean(np.array_split(P, K1 * K2), axis=1).reshape(K1, K2)
    gca_value = Vmat.mean(axis=1) - evalTarget
    IDX = np.sort(gca_value.argsort()[::-1][:K])
    return IDX


def reciprocal_recurrent_selection_fused(ginfo, params, smltr, prdctr, selector):
    GA, GB, RF = ginfo['GA'], ginfo['GB'], ginfo['RF']
    a, dp, dn, pidx, nidx = ginfo['a'], ginfo['dp'], ginfo['dn'], ginfo['pidx'], ginfo['nidx']
    T, H2 = params['T'], params['H2']
    M, K, S, N = params['M'], params['K'], params['S'], GA.shape[0]
    if smltr == "opaque":
        marker = ginfo['marker']
    else:
        marker = None
    phenoA, phenoB, phenoC = np.zeros([N, T]), np.zeros([N, T]), np.zeros([N, T])
    genoA, genoB, genoC = np.zeros([N, T]), np.zeros([N, T]), np.zeros([N, T])
    # transform dominant effect into one vector
    d = dFlat(a, dn, dp, nidx, pidx)
    # initialization for rmse1 and rmse2 and H2Hat for Bayesian predictor
    H2Hat, rmse1, rmse2 = np.zeros([T]), np.zeros([T]), np.zeros([T])
    for t in range(T):
        phenoA[:, t], genoA[:, t] = simulator(GA, a, d, H2)
        phenoB[:, t], genoB[:, t] = simulator(GB, a, d, H2)
        Ghetero = testcross(GA, GB, RF, 1)
        phenoC[:, t] = sorted(simulator(Ghetero, a, d, H2)[0], reverse=True)[:N]
        genoC[:, t] = sorted(simulator(Ghetero, a, d, 1)[1], reverse=True)[:N]

        if prdctr == "Bayesian":
            # G0 = np.concatenate([GA, GB], -0)
            # pheno, _ = simulator(G0, a, d, H2)
            _, genoC_ = simulator(Ghetero, a, d, H2)
            idx = genoC_.argsort()[-5*N:][::-1]
            Ghetero_ = Ghetero[idx, :, :]
            phenoC_, genoC_ = simulator(Ghetero_, a, d, H2)
            if smltr == "transparent":
                # A = pd.DataFrame(np.sum(G0, axis=1) - 1.0)
                # D = pd.DataFrame((np.sum(G0, axis=1) == 1) * 1)
                # gebvCHat, ahat, dhat, S2Hat, H2Hat[t] = blmmp(A, D, pd.DataFrame(pheno))
                A = pd.DataFrame(np.sum(Ghetero_, axis=1) - 1.0)
                D = pd.DataFrame((np.sum(Ghetero_, axis=1) == 1) * 1)
                gebvCHat, ahat, dhat, esigma2, H2Hat[t] = blmmp(A, D, pd.DataFrame(phenoC_))
                # gebvCHat, ahat, dhat, esigma2, H2Hat[t] = blmmp(A, D, pd.DataFrame(genoC_))
                _, evalA = simulator(GA, ahat, dhat, None)
                _, evalB = simulator(GB, ahat, dhat, None)
            if smltr == "opaque":
                # A = pd.DataFrame((np.sum(G0[:, :, marker], axis=1) - 1.0))
                # D = pd.DataFrame((np.sum(G0[:, :, marker], axis=1) == 1) * 1)
                # gebvCHat, ahat, dhat, S2Hat, H2Hat[t] = blmmp(A, D, pd.DataFrame(pheno))
                A = pd.DataFrame(np.sum(Ghetero_[:, :, marker], axis=1) - 1.0)
                D = pd.DataFrame((np.sum(Ghetero_[:, :, marker], axis=1) == 1) * 1)
                gebvCHat, ahat, dhat, esigma2, H2Hat[t] = blmmp(A, D, pd.DataFrame(phenoC_))
                # gebvCHat, ahat, dhat, esigma2, H2Hat[t] = blmmp(A, D, pd.DataFrame(genoC_))
                _, evalA = simulator(GA[:, :, marker], ahat, dhat, None)
                _, evalB = simulator(GB[:, :, marker], ahat, dhat, None)
            # rmse1[t] = np.sqrt(np.nanmean((gebvCHat - genoC_)**2)) / np.mean(genoC_)
            # rmse2[t] = np.sqrt(np.nanmean((gebvCHat - phenoC_)**2)) / np.mean(phenoC_)
            # rmse1[t] = np.sqrt(np.nanmean((gebvCHat - geno) ** 2)) / np.mean(geno)
            # rmse2[t] = np.sqrt(np.nanmean((gebvCHat - pheno) ** 2)) / np.mean(pheno)
        if prdctr == "perfect":
            if smltr == "transparent":
                _, evalA = simulator(GA, a, d, None)
                _, evalB = simulator(GB, a, d, None)
            if smltr == "opaque":
                # _, evalA = simulator(GA[:, :, marker], a[marker], d[marker], None)
                # _, evalB = simulator(GB[:, :, marker], a[marker], d[marker], None)
                _, evalA = simulator(GA, a, d, H2=None)
                _, evalB = simulator(GB, a, d, H2=None)
        if prdctr == "phenotypic":
            if smltr == "transparent":
                evalA, _ = simulator(GA, a, d, H2)
                evalB, _ = simulator(GB, a, d, H2)
            if smltr == "opaque":
                # evalA, _ = simulator(GA[:, :, marker], a[marker], d[marker], H2)
                # evalB, _ = simulator(GB[:, :, marker], a[marker], d[marker], H2)
                evalA, _ = simulator(GA, a, d, H2)
                evalB, _ = simulator(GB, a, d, H2)

        IDXA = evalA.argsort()[::-1][:M]
        IDXB = evalB.argsort()[::-1][:M]

        # random selection in ancillary population
        # Mar 16: remove the random selection block and keep the truncation one
        # IDXa = np.random.permutation([i for i in range(GA.shape[0])])
        # IDXb = np.random.permutation([i for i in range(GB.shape[0])])
        # test cross for selects from population A and B
        # GA2test = testcross(GA[IDXA[:M], :, :], GB[IDXb[:R], :, :], RF, 1)
        # GB2test = testcross(GB[IDXB[:M], :, :], GA[IDXa[:R], :, :], RF, 1)
        GA2test = testcross(GA[IDXA, :, :], GB[IDXB, :, :], RF, 1)
        GB2test = testcross(GB[IDXB, :, :], GA[IDXA, :, :], RF, 1)
        if prdctr == "Bayesian":
            # _, genoC_ = simulator(Ghetero, a, d, H2)
            # idx = genoC_.argsort()[-N:][::-1]
            # Ghetero_ = Ghetero[idx, :, :]
            # phenoC_, genoC_ = simulator(Ghetero_, a, d, H2)
            if smltr == "transparent":
                # A = pd.DataFrame(np.sum(Ghetero_, axis=1) - 1.0)
                # D = pd.DataFrame((np.sum(Ghetero_, axis=1) == 1) * 1)
                # gebvCHat, ahat1, dhat1, esigma2, H2Hat[t] = blmmp(A, D, pd.DataFrame(phenoC_))
                _, evalAtest = simulator(GA2test, ahat, dhat, None)
                _, evalBtest = simulator(GB2test, ahat, dhat, None)
            if smltr == "opaque":
                # A = pd.DataFrame(np.sum(Ghetero_[:, :, marker], axis=1) - 1.0)
                # D = pd.DataFrame((np.sum(Ghetero_[:, :, marker], axis=1) == 1) * 1)
                # gebvCHat, ahat1, dhat1, esigma2, H2Hat[t] = blmmp(A, D, pd.DataFrame(phenoC_))
                _, evalAtest = simulator(GA2test[:, :, marker], ahat, dhat, None)
                _, evalBtest = simulator(GB2test[:, :, marker], ahat, dhat, None)
        if prdctr == "perfect":
            if smltr == "transparent":
                _, evalAtest = simulator(GA2test, a, d, None)
                _, evalBtest = simulator(GB2test, a, d, None)
            if smltr == "opaque":
                _, evalAtest = simulator(GA2test, a, d, H2=None)
                _, evalBtest = simulator(GB2test, a, d, H2=None)
                # _, evalAtest = simulator(GA2test[:, :, marker], a[marker], d[marker], None)
                # _, evalBtest = simulator(GB2test[:, :, marker], a[marker], d[marker], None)
        if prdctr == "phenotypic":
            if smltr == "transparent":
                evalAtest, _ = simulator(GA2test, a, d, H2)
                evalBtest, _ = simulator(GB2test, a, d, H2)
            if smltr == "opaque":
                evalAtest, _ = simulator(GA2test, a, d, H2)
                evalBtest, _ = simulator(GB2test, a, d, H2)
                # evalAtest, _ = simulator(GA2test[:, :, marker], a[marker], d[marker], H2)
                # evalBtest, _ = simulator(GB2test[:, :, marker], a[marker], d[marker], H2)
        # GCA evaluation for test cross
        # Mar 16: keep the original definition for GCA and SCA
        # IDXA2 = IDXA[:K]
        # IDXB2 = IDXB[:K]
        IDXA2 = IDXA[gca(evalAtest, M, M, K)]
        IDXB2 = IDXB[gca(evalBtest, M, M, K)]
        # SCA evaluation
        # IDXA2 = IDXA[sca(evalAtest, M, M, K)]
        # IDXB2 = IDXB[sca(evalBtest, M, M, K)]
        if selector is True:
        # True is complimentary
            # IDXA2_1, IDXA2_2 = IDXA2[:K//2], IDXA2[K//2:]
            # IDXB2_1, IDXB2_2 = IDXB2[:K//2], IDXB2[K//2:]
            IDXA2_ = np.concatenate((IDXA2[::2], IDXA2[1::2]), axis=None)
            IDXA2_1, IDXA2_2 = IDXA2_[::2], IDXA2_[1::2]
            IDXB2_ = np.concatenate((IDXB2[::2], IDXB2[1::2]), axis=None)
            IDXB2_1, IDXB2_2 = IDXB2_[::2], IDXB2_[1::2]
        else:
            IDXA2_1, IDXA2_2 = IDXA2[::2], IDXA2[1::2]
            IDXB2_1, IDXB2_2 = IDXB2[::2], IDXB2[1::2]
        # inter cross within GA fully
        # GA2inter = testcross(GA[IDXA2, :, :], GA[IDXA2, :, :], RF, S)
        # GB2inter = testcross(GB[IDXB2, :, :], GB[IDXB2, :, :], RF, S)
        # selectively cross within GA and GB
        GA2inter = pair_cross(GA[IDXA2_1, :, :], GA[IDXA2_2, :, :], RF, 1)
        GB2inter = pair_cross(GB[IDXB2_1, :, :], GB[IDXB2_2, :, :], RF, 1)
        # elites for haploid duplication
        if prdctr == "Bayesian":
            if smltr == "transparent":
                _, evalAHD = simulator(GA2inter, ahat, dhat, None)
                _, evalBHD = simulator(GB2inter, ahat, dhat, None)
            if smltr == "opaque":
                _, evalAHD = simulator(GA2inter[:, :, marker], ahat, dhat, None)
                _, evalBHD = simulator(GB2inter[:, :, marker], ahat, dhat, None)
        if prdctr == "perfect":
            if smltr == "transparent":
                _, evalAHD = simulator(GA2inter, a, d, None)
                _, evalBHD = simulator(GB2inter, a, d, None)
            if smltr == "opaque":
                _, evalAHD = simulator(GA2inter, a, d, H2=None)
                _, evalBHD = simulator(GB2inter, a, d, H2=None)
                # _, evalAHD = simulator(GA3[:, :, marker], a[marker], d[marker], None)
                # _, evalBHD = simulator(GB3[:, :, marker], a[marker], d[marker], None)
        if prdctr == "phenotypic":
            if smltr == "transparent":
                evalAHD, _ = simulator(GA2inter, a, d, H2)
                evalBHD, _ = simulator(GB2inter, a, d, H2)
            if smltr == "opaque":
                evalAHD, _ = simulator(GA2inter, a, d, H2)
                evalBHD, _ = simulator(GB2inter, a, d, H2)
                # evalAHD, _ = simulator(GA3[:, :, marker], a[marker], d[marker], H2)
                # evalBHD, _ = simulator(GB3[:, :, marker], a[marker], d[marker], H2)
        IDXA = evalAHD.argsort()[::-1][:(K//4)]
        IDXB = evalBHD.argsort()[::-1][:(K//4)]
        GA2_ = GA2inter[IDXA, :, :]
        GB2_ = GB2inter[IDXB, :, :]
        n1, n2, L = GA2_.shape[0], GB2_.shape[0], GA2_.shape[2]
        GA3 = reproduce(GA2_[np.repeat(range(n1), 2), :, :], RF, np.ones(n1, dtype=int) * S)
        GB3 = reproduce(GB2_[np.repeat(range(n2), 2), :, :], RF, np.ones(n2, dtype=int) * S)
        # Apr 9: add the process for doubled haploid: one individual produces one gamete after recombination
        # which is equivalent to self cross again in codes and this will give two gametes one time
        # Magic number here
        # S1 = 2
        # n1, n2, L = GA3.shape[0], GB3.shape[0], GA3.shape[2]
        # GA3_ = GA3[IDXA, :, :]
        # GB3_ = GB3[IDXB, :, :]
        # gA3 = reproduce(GA3_[np.repeat(range(n1), 2), :, :], RF, np.ones(n1, dtype=int) * S1)
        # gB3 = reproduce(GB3_[np.repeat(range(n2), 2), :, :], RF, np.ones(n2, dtype=int) * S1)
        # broadcast an individual into two gametes for each
        gA3 = GA3.reshape(2 * n1 * S, 1, L)
        gB3 = GB3.reshape(2 * n2 * S, 1, L)
        IDXa = np.random.permutation(np.arange(2 * n1 * S))[:N]
        IDXb = np.random.permutation(np.arange(2 * n2 * S))[:N]
        gA = gA3[IDXa, :, :]
        gB = gB3[IDXb, :, :]
        # Doubled haploid
        GA = np.tile(gA, (2, 1))
        GB = np.tile(gB, (2, 1))
    print("Finished one loop. \n")
    return genoA, genoB, genoC, phenoA, phenoB, phenoC, H2Hat, rmse1, rmse2


def rrs(ginfo, params, smltr, prdctr, selector, B, seed=None):
    if seed is None:
        seed = 1001
    T = params['T']
    GAcmplt = ginfo['GA']
    GBcmplt = ginfo['GB']
    nA = GAcmplt.shape[0]
    nB = GBcmplt.shape[0]
    g = copy.deepcopy(ginfo)
    N = 100
    genoA, genoB, genoC = np.zeros(shape=(B * N, T)), np.zeros(shape=(B * N, T)), np.zeros(shape=(B * N, T))
    phenoA, phenoB, phenoC = np.zeros(shape=(B * N, T)), np.zeros(shape=(B * N, T)), np.zeros(shape=(B * N, T))
    H2Hat, rmse1, rmse2 = np.zeros(shape=(B, T)), np.zeros(shape=(B, T)), np.zeros(shape=(B, T))
    for i in range(B):
        np.random.seed(seed + i)
        idx1 = np.random.choice(nA, N, replace=False)
        idx2 = np.random.choice(nB, N, replace=False)
        seed += 1
        g['GA'] = GAcmplt[idx1, :, :]
        g['GB'] = GBcmplt[idx2, :, :]
        if prdctr != "Bayesian":
            genoA[i*N:(i+1)*N, :], genoB[i*N:(i+1)*N, :], genoC[i*N:(i+1)*N, :], phenoA[i*N:(i+1)*N, :], \
            phenoB[i*N:(i+1)*N, :], phenoC[i*N:(i+1)*N, :], _, _, _ = reciprocal_recurrent_selection_fused(g, params, smltr, prdctr, selector)
        else:
            genoA[i * N:(i + 1) * N, :], genoB[i * N:(i + 1) * N, :], genoC[i * N:(i + 1) * N, :], phenoA[i * N:(i + 1) * N,:], \
            phenoB[i * N:(i + 1) * N, :], phenoC[i * N:(i + 1) * N, :], H2Hat[i, :], rmse1[i, :], rmse2[i, :] = reciprocal_recurrent_selection_fused(
                g, params, smltr, prdctr, selector)
    y = np.hstack([np.mean(genoA, axis=0).reshape(1, T),
                   np.mean(genoB, axis=0).reshape(1, T),
                   np.mean(genoC, axis=0).reshape(1, T),
                   np.mean(phenoA, axis=0).reshape(1, T),
                   np.mean(phenoB, axis=0).reshape(1, T),
                   np.mean(phenoC, axis=0).reshape(1, T)])
    df = pd.DataFrame(np.c_[np.tile(np.repeat(['A', 'B', 'C'], T), 2),
                            np.repeat(['Genotype', 'Phenotype'], T * 3),
                            y.squeeze(),
                            np.tile(range(T), 2 * 3),
                            np.tile(smltr, T * 2 * 3),
                            np.tile(prdctr, T * 2 * 3)],
                      columns=["Population", "Response", "Value", "Generation", "Simulator", "Predictor"])
    df["Generation"] = pd.to_numeric(df["Generation"])
    df["Value"] = pd.to_numeric(df["Value"])
    evaluation = pd.DataFrame({
        'Predictor': np.tile(smltr, T * 3),
        'Simulator': np.tile(prdctr, T * 3),
        'Generation': np.tile(range(0, T), 3),
        'Population': np.repeat(['A', 'B', "C"], T)}
    )
    return df, evaluation, genoA, genoB, genoC, phenoA, phenoB, phenoC, pd.DataFrame(H2Hat), pd.DataFrame(rmse1), pd.DataFrame(rmse2)


def adaptiveA(G, aHat, dHat):
    dHatp = dHat[dHat > 0]
    dHatn = dHat[dHat < 0]
    pidxHat = np.where(dHat > 0)
    nidxHat = np.where(dHat < 0)

    _, GEBV = simulator(G, aHat, dHat, None)
    bonus = np.matmul(np.sum(G[:, :, pidxHat], axis=1) == 2, dHatp).flatten()
    penalty = np.matmul(np.sum(G[:, :, nidxHat], axis=1) == 2, dHatn).flatten()
    GEBV = GEBV + bonus - penalty
    return GEBV


def adaptiveB(G, aHat, dHat):
    dHatp = dHat[dHat > 0]
    dHatn = dHat[dHat < 0]
    pidxHat = np.where(dHat > 0)
    nidxHat = np.where(dHat < 0)

    _, GEBV = simulator(G, aHat, dHat, None)
    bonus = np.matmul(np.sum(G[:, :, pidxHat], axis=1) == 0, dHatp).flatten()
    penalty = np.matmul(np.sum(G[:, :, nidxHat], axis=1) == 2, dHatn).flatten()
    GEBV = GEBV + bonus - penalty
    return GEBV


def hyp_intercross(G, RF, W):
    n = G.shape[0]
    if n % 2 != 0:
        raise Exception("Genome dimension cannot intercross")
    tic = 1
    G_ = G
    while tic <= W:
        nps = np.ones(n // 2, dtype=int) * 2
        if tic <= 2:
            G_ = reproduce(G_[range(n), :, :], RF, nps)
        else:
            idx = np.random.permutation(np.arange(n))
            G_ = reproduce(G_[idx, :, :], RF, nps)
        tic += 1
    return G_


def bayesian_adaptive_selection(ginfo, params, smltr, prdctr, selector):
    GA, GB, RF = ginfo['GA'], ginfo['GB'], ginfo['RF']
    a, dp, dn, pidx, nidx = ginfo['a'], ginfo['dp'], ginfo['dn'], ginfo['pidx'], ginfo['nidx']
    T, H2 = params['T'], params['H2']
    M, K, S, N = params['M'], params['K'], params['S'], GA.shape[0]
    if smltr == "opaque":
        marker = ginfo['marker']
    else:
        marker = None
    phenoA, phenoB, phenoC = np.zeros([N, T]), np.zeros([N, T]), np.zeros([N, T])
    genoA, genoB, genoC = np.zeros([N, T]), np.zeros([N, T]), np.zeros([N, T])
    # transform dominant effect into one vector
    d = dFlat(a, dn, dp, nidx, pidx)
    for t in range(T):
        phenoA[:, t], genoA[:, t] = simulator(GA, a, d, H2)
        phenoB[:, t], genoB[:, t] = simulator(GB, a, d, H2)
        Ghetero = testcross(GA, GB, RF, 1)
        phenoC[:, t] = sorted(simulator(Ghetero, a, d, H2)[0], reverse=True)[:N]
        genoC[:, t] = sorted(simulator(Ghetero, a, d, 1)[1], reverse=True)[:N]

        if prdctr == "Bayesian":
            # G0 = np.concatenate([GA, GB], -0)
            # _, pheno = simulator(G0, a, d, H2)
            _, genoC_ = simulator(Ghetero, a, d, H2)
            idx = genoC_.argsort()[-N:][::-1]
            Ghetero_ = Ghetero[idx, :, :]
            phenoC_, genoC_ = simulator(Ghetero_, a, d, H2)
            if smltr == "transparent":
                # A = pd.DataFrame(np.sum(G0, axis=1) - 1.0)
                # D = pd.DataFrame((np.sum(G0, axis=1) == 1) * 1)
                # gebvCHat, ahat, dhat, _, H2Hat[t] = blmmp(A, D, pd.DataFrame(pheno))
                A = pd.DataFrame(np.sum(Ghetero_, axis=1) - 1.0)
                D = pd.DataFrame((np.sum(Ghetero_, axis=1) == 1) * 1)
                gebvCHat, ahat, dhat, esigma2, _ = blmmp(A, D, pd.DataFrame(phenoC_))
                evalA = adaptiveA(GA, ahat, dhat)
                evalB = adaptiveB(GB, ahat, dhat)
            if smltr == "opaque":
                # A = pd.DataFrame((np.sum(G0[:, :, marker], axis=1) - 1.0))
                # D = pd.DataFrame((np.sum(G0[:, :, marker], axis=1) == 1) * 1)
                # gebvCHat, ahat, dhat, _, H2Hat[t] = blmmp(A, D, pd.DataFrame(pheno))
                A = pd.DataFrame(np.sum(Ghetero_[:, :, marker], axis=1) - 1.0)
                D = pd.DataFrame((np.sum(Ghetero_[:, :, marker], axis=1) == 1) * 1)
                gebvCHat, ahat, dhat, esigma2, _ = blmmp(A, D, pd.DataFrame(phenoC_))
                evalA = adaptiveA(GA[:, :, marker], ahat, dhat)
                evalB = adaptiveB(GB[:, :, marker], ahat, dhat)

        IDXA = evalA.argsort()[::-1][:M]
        IDXB = evalB.argsort()[::-1][:M]
        while True:
            GA2test = testcross(GA[IDXA, :, :], GB[IDXB, :, :], RF, 1)
            GB2test = testcross(GB[IDXB, :, :], GA[IDXA, :, :], RF, 1)
            if prdctr == "Bayesian":
                if smltr == "transparent":
                    _, evalAtest = simulator(GA2test, ahat, dhat, None)
                    _, evalBtest = simulator(GB2test, ahat, dhat, None)
                if smltr == "opaque":
                    _, evalAtest = simulator(GA2test[:, :, marker], ahat, dhat, None)
                    _, evalBtest = simulator(GB2test[:, :, marker], ahat, dhat, None)
            IDXA2 = IDXA[gca(evalAtest, M, M, K)]
            IDXB2 = IDXB[gca(evalBtest, M, M, K)]

        # SCA evaluation
        # IDXA2 = IDXA[sca(evalAtest, M, M, K)]
        # IDXB2 = IDXB[sca(evalBtest, M, M, K)]
        if selector is True:
        # True is complimentary
            IDXA2_1, IDXA2_2 = IDXA2[:K//2], IDXA2[K//2:]
            IDXB2_1, IDXB2_2 = IDXB2[:K//2], IDXB2[K//2:]
        else:
            IDXA2_1, IDXA2_2 = IDXA2[::2], IDXA2[1::2]
            IDXB2_1, IDXB2_2 = IDXB2[::2], IDXB2[1::2]
        # inter cross within GA fully
        # GA2inter = testcross(GA[IDXA2, :, :], GA[IDXA2, :, :], RF, S)
        # GB2inter = testcross(GB[IDXB2, :, :], GB[IDXB2, :, :], RF, S)
        # selectively cross within GA and GB
        GA2inter = pair_cross(GA[IDXA2_1, :, :], GA[IDXA2_2, :, :], RF, 1)
        GB2inter = pair_cross(GB[IDXB2_1, :, :], GB[IDXB2_2, :, :], RF, 1)
        # elites for haploid duplication
        if prdctr == "Bayesian":
            if smltr == "transparent":
                _, evalAHD = simulator(GA2inter, ahat, dhat, None)
                _, evalBHD = simulator(GB2inter, ahat, dhat, None)
            if smltr == "opaque":
                _, evalAHD = simulator(GA2inter[:, :, marker], ahat, dhat, None)
                _, evalBHD = simulator(GB2inter[:, :, marker], ahat, dhat, None)

        IDXA = evalAHD.argsort()[::-1][:(K//4)]
        IDXB = evalBHD.argsort()[::-1][:(K//4)]
        GA2_ = GA2inter[IDXA, :, :]
        GB2_ = GB2inter[IDXB, :, :]
        n1, n2, L = GA2_.shape[0], GB2_.shape[0], GA2_.shape[2]
        GA3 = reproduce(GA2_[np.repeat(range(n1), 2), :, :], RF, np.ones(n1, dtype=int) * S)
        GB3 = reproduce(GB2_[np.repeat(range(n2), 2), :, :], RF, np.ones(n2, dtype=int) * S)
        # broadcast an individual into two gametes for each
        gA3 = GA3.reshape(2 * n1 * S, 1, L)
        gB3 = GB3.reshape(2 * n2 * S, 1, L)
        IDXa = np.random.permutation(np.arange(2 * n1 * S))[:N]
        IDXb = np.random.permutation(np.arange(2 * n2 * S))[:N]
        gA = gA3[IDXa, :, :]
        gB = gB3[IDXb, :, :]
        # Doubled haploid
        GA = np.tile(gA, (2, 1))
        GB = np.tile(gB, (2, 1))
    print("Finished one loop. \n")
    return genoA, genoB, genoC, phenoA, phenoB, phenoC


def rrs_adaptive(ginfo, params, smltr, prdctr, selector, B, seed=None):
    if seed is None:
        seed = 1001
    T = params['T']
    GAcmplt = ginfo['GA']
    GBcmplt = ginfo['GB']
    nA = GAcmplt.shape[0]
    nB = GBcmplt.shape[0]
    g = copy.deepcopy(ginfo)
    N = 100
    genoA, genoB, genoC = np.zeros(shape=(B * N, T)), np.zeros(shape=(B * N, T)), np.zeros(shape=(B * N, T))
    phenoA, phenoB, phenoC = np.zeros(shape=(B * N, T)), np.zeros(shape=(B * N, T)), np.zeros(shape=(B * N, T))
    H2Hat, rmse1, rmse2 = np.zeros(shape=(B, T)), np.zeros(shape=(B, T)), np.zeros(shape=(B, T))
    for i in range(B):
        np.random.seed(seed + i)
        idx1 = np.random.choice(nA, N, replace=False)
        idx2 = np.random.choice(nB, N, replace=False)
        seed += 1
        g['GA'] = GAcmplt[idx1, :, :]
        g['GB'] = GBcmplt[idx2, :, :]
        genoA[i * N:(i + 1) * N, :], genoB[i * N:(i + 1) * N, :], genoC[i * N:(i + 1) * N, :], phenoA[i * N:(i + 1) * N,:], \
        phenoB[i * N:(i + 1) * N, :], phenoC[i * N:(i + 1) * N, :] = bayesian_adaptive_selection(g, params, smltr, prdctr, selector)
    y = np.hstack([np.mean(genoA, axis=0).reshape(1, T),
                   np.mean(genoB, axis=0).reshape(1, T),
                   np.mean(genoC, axis=0).reshape(1, T),
                   np.mean(phenoA, axis=0).reshape(1, T),
                   np.mean(phenoB, axis=0).reshape(1, T),
                   np.mean(phenoC, axis=0).reshape(1, T)])
    df = pd.DataFrame(np.c_[np.tile(np.repeat(['A', 'B', 'C'], T), 2),
                            np.repeat(['Genotype', 'Phenotype'], T * 3),
                            y.squeeze(),
                            np.tile(range(T), 2 * 3),
                            np.tile(smltr, T * 2 * 3),
                            np.tile(prdctr, T * 2 * 3)],
                      columns=["Population", "Response", "Value", "Generation", "Simulator", "Predictor"])
    df["Generation"] = pd.to_numeric(df["Generation"])
    df["Value"] = pd.to_numeric(df["Value"])
    evaluation = pd.DataFrame({
        'Predictor': np.tile(smltr, T * 3),
        'Simulator': np.tile(prdctr, T * 3),
        'Generation': np.tile(range(0, T), 3),
        'Population': np.repeat(['A', 'B', "C"], T)}
    )
    return df, evaluation, genoA, genoB, genoC, phenoA, phenoB, phenoC