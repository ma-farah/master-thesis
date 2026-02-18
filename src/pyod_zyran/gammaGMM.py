import numpy as np
import pandas as pd
from scipy.stats import beta, wishart, multivariate_normal, dirichlet
from sklearn.mixture import BayesianGaussianMixture
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.iforest import IForest
from scipy.optimize import least_squares
from multiprocessing import Pool, cpu_count
from functools import partial


def run_gammaGMM(
    X,
    ad_list=[KNN(), IForest(), LOF()],
    tot_samples=10000,
    ndraws=100,
    p0=0.01,
    phigh=0.01,
    high_gamma=0.15,
    gamma_lim=0.25,
    K=100,
    seed=331,
    cpu=1,
    verbose=True,
):
    """
    Estimate the contamination factor's posterior distribution following the algorithm provided in the paper.
    First, it uses the ad_list of detectors to assign the anomaly scores (detectors implemented in PyOD).
    Second, it fits a Dirichlet Process Gaussian Mixture Model on the scores.
    Third, it finds the optimal hyperparameters using the input p0 and phigh values.
    Fourth, it computes the (joint) probability that k components are anomalous.
    Finally, it derives a sample from the posterior distribution of size = tot_samples.

    Parameters
    ----------
    X              : np.array of shape (n,m) containing the training set;
    ad_list        : list of shape (M,) containing the anomaly detectors as implemented in PyOD;
    tot_samples    : integer that refers to the sample size drawn from gamma's posterior;
    ndraws         : integer that refers to the number of samples drawn from each component's mean/std sample;
    p0,phigh       : float that refers to the hyperparameter p0 or phigh as in the paper;
    high_gamma     : float that indicates the point such that the probability of obtaining higher gammas is phigh;
    gamma_lim      : float that limits the values of gamma;
    K              : integer with the upper limit the number of components for the DPGMM;
    seed           : integer value for reproducibility of the experiments;
    cpu            : number of cpus that can be used to parallelize the code;
    verbose        : True if you want to check the status of the DPGMM every 20 iterations, False otherwise.

    Returns
    ----------
    gamma          : np.array of shape (tot_samples,) with the samples drawn from gamma's posterior distribution.

    """

    # we cannot use more components than samples.
    if np.shape(X)[0] < K:
        K = np.shape(X)[0] - 1

    itv = 0
    # Repeat the loop until a valid gamma sample is found. If it cannot, then return gamma = 0.
    while True:
        whileseed = seed + 100 * itv
        np.random.seed(whileseed)
        # Compute the anomaly scores -> Pool is needed for parallelization
        pool = Pool(min(cpu, cpu_count()))
        dflist = pool.map(partial(get_anomaly_scores, X=X), ad_list)
        pool.close()
        pool.join()
        df = pd.concat(dflist, axis=1)
        scores = df.values
        # Fit the DPGMM
        bgm = BayesianGaussianMixture(
            weight_concentration_prior_type="dirichlet_process",
            n_components=K,
            weight_concentration_prior=0.01,
            max_iter=1500,
            random_state=whileseed,
            verbose=verbose,
            verbose_interval=20,
        ).fit(scores)
        # Drop components with less than 2 instances assigned
        filter_idx = np.where(bgm.weight_concentration_[0] >= 2)[0]
        tot_concentration = np.sum(bgm.weight_concentration_[0])
        partial_concentration = np.sum(bgm.weight_concentration_[0][filter_idx])
        means = bgm.means_[filter_idx]
        covariances = bgm.covariances_[filter_idx, :, :]
        alphas = bgm.weight_concentration_[0][filter_idx]
        mean_precs = bgm.mean_precision_[filter_idx]
        dgf = bgm.degrees_of_freedom_[filter_idx]
        idx_sortcomponents, meanstd = order_components(
            means, mean_precs, covariances, dgf, whileseed
        )
        # Redistribute the lost mass (after cutting off some components)
        alphas = alphas[idx_sortcomponents] + (
            tot_concentration - partial_concentration
        ) / len(filter_idx)
        # Solve the optimization problem to find the hyperparameters delta and tau
        res = least_squares(
            find_delta_tau,
            x0=(-2, 1),
            args=(meanstd, alphas, high_gamma, p0, phigh),
            bounds=((-50, -1), (0, 50)),
        )
        delta, tau = res.x
        # Check that delta and tau are properly found, allowing 10% error top
        p0Est, phighEst = check_delta_tau(delta, tau, meanstd, alphas, high_gamma)
        if (
            p0Est < p0 * 1.10
            and p0Est > p0 * 0.90
            and phighEst < phigh * 1.10
            and phighEst > phigh * 0.90
        ):
            break
        elif verbose:
            print("It cannot find the best hyperparameters. Let's run the model again.")
        itv += 1
        if itv > 100:
            print("No solution found. It will return all zeros.")
            return np.zeros(tot_samples, float)
    # Sort the components values and extract the parameters posteriors
    means = means[idx_sortcomponents]
    mean_precs = bgm.mean_precision_[idx_sortcomponents]
    covariances = covariances[idx_sortcomponents, :, :]
    dgf = bgm.degrees_of_freedom_[idx_sortcomponents]
    # Compute the cumulative sum of the mixing proportion (GMM weights)
    gmm_weights = np.cumsum(dirichlet(alphas).rvs(tot_samples), axis=1)
    w = {}
    for k in range(len(filter_idx)):
        w[k + 1] = gmm_weights[:, k]
    # Sample from gamma's posterior by computing the probabilities
    gamma = sample_withexactprobs(
        means,
        mean_precs,
        covariances,
        dgf,
        delta,
        tau,
        p0,
        w,
        tot_samples=tot_samples,
        ndraws=ndraws,
        seed=whileseed,
    )

    # if gamma has not enough samples, just do oversampling
    gamma = gamma[gamma < gamma_lim]
    if len(gamma) < tot_samples:
        gamma = np.concatenate(
            (
                gamma,
                np.random.choice(
                    gamma[gamma > 0.0], tot_samples - len(gamma), replace=True
                ),
            )
        )
    return gamma


def order_components(means, mean_precs, covariances, dgf, seed):
    K, M = np.shape(means)
    meanstd = np.zeros(K, float)
    mean_std = np.sqrt(1 / mean_precs)
    for k in range(K):
        sample_mean_component = multivariate_normal.rvs(
            mean=means[k, :], cov=mean_std[k] ** 2, size=1000, random_state=seed
        )
        sample_covariance = wishart.rvs(
            df=dgf[k], scale=covariances[k] / dgf[k], size=1000, random_state=seed
        )
        var = np.array([np.diag(sample_covariance[i]) for i in range(1000)])
        meanstd[k] = np.mean(
            [
                np.mean(
                    sample_mean_component[:, m].reshape(-1)
                    / (1 + np.sqrt(var[:, m].reshape(-1)))
                )
                for m in range(M)
            ]
        )
    idx_components = np.argsort(-meanstd)
    meanstd = meanstd[idx_components]
    return idx_components, np.array(meanstd)


def find_delta_tau(params, *args):
    delta, tau = params
    meanstd, alphas, high_gamma, p0, phigh = args
    first_eq = delta - (np.log(p0 / (1 - p0)) - tau) / meanstd[0]

    prob_ck = sigmoid(delta, tau, meanstd)
    prob_c1ck = derive_jointprobs(prob_ck)

    a = np.cumsum(alphas)
    b = sum(alphas) - np.cumsum(alphas)
    probBetaGreaterT = np.nan_to_num(beta.sf(high_gamma, a, b), nan=1.0)

    second_eq = np.sum(probBetaGreaterT * prob_c1ck) - phigh

    return (first_eq, second_eq)


def check_delta_tau(delta, tau, meanstd, alphas, high_gamma):
    prob_ck = sigmoid(delta, tau, meanstd)
    p0Est = 1 - prob_ck[0]

    prob_c1ck = derive_jointprobs(prob_ck)
    a = np.cumsum(alphas)
    b = sum(alphas) - np.cumsum(alphas)
    probBetaGreaterT = np.nan_to_num(beta.sf(high_gamma, a, b), nan=1.0)
    phighEst = np.sum(probBetaGreaterT * prob_c1ck)
    return p0Est, phighEst


def sigmoid(delta, tau, x):
    return 1 / (1 + np.exp(tau + delta * x))


def derive_jointprobs(prob_ck):
    cumprobs = np.cumprod(prob_ck)
    negprobs = np.roll(1 - prob_ck, -1)
    negprobs[-1] = 1
    prob_c1ck = cumprobs * negprobs
    return prob_c1ck


def sample_withexactprobs(
    means,
    mean_precs,
    covariances,
    dgf,
    delta,
    tau,
    p0,
    w,
    tot_samples=10000,
    ndraws=100,
    seed=331,
):
    K = np.shape(means)[0]
    mean_std = np.sqrt(1 / mean_precs)
    samples = np.array([])
    i = 0
    while len(samples) < tot_samples * (1 - p0):
        prob_ck = np.zeros(K, float)
        for k in range(K):
            rnd = (i + 1) * (k + 1)
            sample_mean_component = multivariate_normal.rvs(
                mean=means[k, :],
                cov=mean_std[k] ** 2,
                size=1,
                random_state=10 * seed + rnd,
            )
            sample_covariance = wishart.rvs(
                df=dgf[k],
                scale=covariances[k] / dgf[k],
                size=1,
                random_state=10 * seed + rnd,
            )
            var = np.diag(sample_covariance)
            meanstd = np.mean((sample_mean_component) / (1 + np.sqrt(var)))

            prob_ck[k] = sigmoid(delta, tau, meanstd)

        prob_c1ck = derive_jointprobs(prob_ck)
        for k in range(K):
            ns = int(np.round(ndraws * prob_c1ck[k], 0))
            if ns > 0:
                samples = np.concatenate(
                    (samples, np.random.choice(w[k + 1], ns, replace=False))
                )
        i += 1
    if len(samples) > tot_samples * (1 - p0):
        samples = np.random.choice(samples, int(tot_samples * (1 - p0)), replace=False)
    samples = np.concatenate((samples, np.zeros(tot_samples - len(samples), float)))
    return samples


def get_anomaly_scores(detector, X):
    detector.fit(X)
    x = detector.decision_function(X)
    minx = min(x)
    x = np.log(x - minx + 0.01)
    meanx, stdx = np.mean(x), np.std(x)
    if stdx == 0:
        print(
            "Detector",
            detector,
            "assigns constant scores (unique value). Removing this detector is suggested.",
        )
    else:
        x = (x - meanx) / stdx
    df = pd.DataFrame(data=x)
    return df
