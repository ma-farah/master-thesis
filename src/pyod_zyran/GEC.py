import numpy as np
import scipy.special as special
from itertools import combinations_with_replacement as co
from itertools import combinations
import pandas as pd


def calculate_train_scores(X, list_OD_algorithms, list_OD_algorithms_string):
    train_scores = np.zeros([X.shape[0], len(list_OD_algorithms)])

    for i in range(len(list_OD_algorithms)):
        clf = list_OD_algorithms[i]
        clf.fit(X)

        train_scores[:, i] = clf.decision_scores_

    train_scores = pd.DataFrame(train_scores)
    train_scores.columns = list_OD_algorithms_string

    return train_scores


def calculate_GEC(X, list_OD_algorithms, list_OD_algorithms_string, percentages):
    train_scores = calculate_train_scores(
        X, list_OD_algorithms, list_OD_algorithms_string
    )

    tau_tot_df = pd.DataFrame(
        index=list_OD_algorithms_string, columns=list_OD_algorithms_string
    )

    for i, j in combinations(train_scores.columns, 2):
        score1 = np.array(train_scores[i])
        score2 = np.array(train_scores[j])

        length = score1.shape[0]
        PERCENTAGES = percentages
        BORDERS = [
            (
                round(length * PERCENTAGES[i - 1] if i != 0 else 0),
                round(length * PERCENTAGES[i]),
            )
            for i in range(len(PERCENTAGES))
        ]

        ord_sco1 = np.argsort(score1)
        index = np.arange(score1.size)

        score1 = np.concatenate((score1[ord_sco1, np.newaxis], index[:, np.newaxis]), 1)
        score2 = np.concatenate((score2[ord_sco1, np.newaxis], index[:, np.newaxis]), 1)

        scores = [score1, score2]

        def score_preprocessing(scores):
            ord_scores = []
            for sc in scores:
                ord_scores.append(sc[np.argsort(sc[:, 0]), :])
            classes = [[] for i in range(2)]
            for j in range(len(BORDERS)):
                for i, ord_s in enumerate(ord_scores):
                    classes[i].append(ord_s[BORDERS[j][0] : BORDERS[j][1]])
            means = [[] for _ in range(len(BORDERS))]
            stds = [[] for _ in range(len(BORDERS))]
            for j in range(len(BORDERS)):
                c_i = set(classes[0][j][:, 1]).intersection(set(classes[1][j][:, 1]))
                for k in range(2):
                    mean = np.mean(classes[k][j][:, 0])
                    means[j].append(mean)
                    std = np.std(classes[k][j][:, 0])
                    stds[j].append(std)
                    arr = classes[k][j]
                    inds = np.asarray(list(set(arr[:, 1]) - c_i), dtype=int)
                    classes[k][j] = arr[np.isin(arr[:, 1], inds)]
            return means, stds, classes

        def reindexing(classes):
            state = 0
            keys = []
            vals = []
            for i in range(len(BORDERS)):
                key = classes[0][i][:, 1]
                val = np.arange(state, state + key.size)
                state = state + key.size
                keys.extend(key)
                vals.extend(val)
                classes[0][i][:, 1] = val
            my_dict = dict(zip(keys, vals))
            for j in range(len(BORDERS)):
                new_keys = classes[1][j][:, 1]
                new_vals = list(map(my_dict.get, new_keys))
                if None in new_vals:
                    missing_keys = [
                        key for key, val in zip(new_keys, new_vals) if val is None
                    ]
                    raise ValueError(f"Missing keys: {missing_keys}")
                classes[1][j][:, 1] = new_vals

            return len(vals), classes

        def gaussian_ev(points, mu, std):
            arg = (points - mu) / (std * np.sqrt(2))
            fx = 1 / 2 * (1 + special.erf(arg))
            return fx

        def integral_weights(ps, mus, stds):
            weights = 0
            for mu, std in zip(mus, stds):
                dists = []
                for point in ps:
                    dists.append(gaussian_ev(point, mu, std))
                weight = np.abs(np.subtract.outer(dists[0], dists[1]))
                weights += weight
            return weights

        def ind_sign(ps, same):
            ratio = np.divide.outer(ps[0] + 1, ps[1] + 1)
            if same == False:
                rat = np.where(ratio > 1, -2, 2)
            else:
                rat = np.where(ratio > 1, -1, 1)
            return rat

        def tau(means, stds, classes, length):
            coeff = np.zeros((length, length, 2))
            sets = list(range(len(BORDERS)))
            for k in range(2):
                clas = classes[k]
                for i, j in co(sets, 2):
                    offset = 0
                    if i - j > 1:
                        offset = 1
                    ps = [clas[i][:, 0], clas[j][:, 0]]
                    mus = [means[i][k], means[j][k]]
                    sta = [stds[i][k], stds[j][k]]
                    ws = integral_weights(ps, mus, sta) + offset
                    in_t = [clas[i][:, 1], clas[j][:, 1]]
                    same = False
                    if i == j:
                        same = True
                    ratios = ind_sign(ps, same)
                    xc = np.repeat(in_t[0][:, np.newaxis], in_t[1].size, axis=1)
                    yc = np.repeat(in_t[1][np.newaxis, :], in_t[0].size, axis=0)
                    xc = xc.astype(int).flatten()
                    yc = yc.astype(int).flatten()
                    part_coef = ws * ratios
                    coeff[xc, yc, k] = part_coef.flatten()
            return coeff

        def coef_prep(coef):
            for i in range(2):
                coef[:, :, i] = np.where(
                    coef[:, :, i] != 0, coef[:, :, i], -np.transpose(coef[:, :, i])
                )
                coef[:, :, i] = np.triu(coef[:, :, i], 1)
            return coef

        def score_computing(mus, sigmas, classes, length):
            coef = tau(mus, sigmas, classes, length)
            coef = coef_prep(coef)
            tau_s = np.sum(coef[:, :, 0] * coef[:, :, 1])
            return tau_s, coef

        mus, sigmas, classes = score_preprocessing(scores)

        length, classes = reindexing(classes)

        mu_norm1 = [[mus[j][0] for i in range(2)] for j in range(len(BORDERS))]
        mu_norm2 = [[mus[j][1] for i in range(2)] for j in range(len(BORDERS))]
        si_norm1 = [[sigmas[j][0] for i in range(2)] for j in range(len(BORDERS))]
        si_norm2 = [[sigmas[j][1] for i in range(2)] for j in range(len(BORDERS))]
        class_norm1 = [[classes[0][i] for i in range(len(BORDERS))] for j in range(2)]
        class_norm2 = [[classes[1][i] for i in range(len(BORDERS))] for j in range(2)]

        tau_s1, coef_s1 = score_computing(mu_norm1, si_norm1, class_norm1, length)
        tau_co, coef_co = score_computing(mus, sigmas, classes, length)
        tau_s2, coef_s2 = score_computing(mu_norm2, si_norm2, class_norm2, length)

        tau_tot = tau_co / (np.sqrt(tau_s1 * tau_s2))
        tau_tot_df.loc[j, i] = tau_tot
        tau_tot_df.loc[i, j] = tau_tot


    # Set the diagonal to NaN since it's 0 and we don't want to include it in our calculations
    np.fill_diagonal(tau_tot_df.values, np.nan)

    # Melting the DataFrame to make it easier to sort and filter
    melted_df = tau_tot_df.melt(ignore_index=False).reset_index()
    melted_df.columns = ['Algorithm1', 'Algorithm2', 'Dissimilarity']
    melted_df = melted_df.dropna()

    # Sorting the dissimilarity scores in ascending order, since values close to -1 indicate higher dissimilarity
    sorted_dissimilar_pairs = melted_df.sort_values(by='Dissimilarity', ascending=True)

    # Create a list of unique algorithms, prioritising the most dissimilar pairs
    unique_algorithms_list = []
    for index, row in sorted_dissimilar_pairs.iterrows():
        if row['Algorithm1'] not in unique_algorithms_list:
            unique_algorithms_list.append(row['Algorithm1'])
        if row['Algorithm2'] not in unique_algorithms_list:
            unique_algorithms_list.append(row['Algorithm2'])
        if len(unique_algorithms_list) >= 6:
            break

    final_selected_algorithms = unique_algorithms_list[:6]

    return final_selected_algorithms, tau_tot_df
