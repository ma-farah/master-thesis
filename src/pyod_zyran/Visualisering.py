import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from adjustText import adjust_text
import hoggorm as ho
import hoggormplot as hop
import os


def prepare_data(X_data_to_use):
    if isinstance(X_data_to_use, pd.DataFrame):
        data = X_data_to_use.values
        data_varNames = list(X_data_to_use.columns)
    elif isinstance(X_data_to_use, np.ndarray):
        data = X_data_to_use
        data_varNames = list(range(X_data_to_use.shape[1]))
    else:
        raise TypeError("X_data_to_use must be a pandas DataFrame or a numpy array.")
    return data, data_varNames


def detect_outliers(X_data_to_use, X_sc, list_OD_algorithms, patient_ids):
    y_predproba_array = np.zeros([X_data_to_use.shape[0], len(list_OD_algorithms)])
    y_pred_confidence_array = np.zeros([X_data_to_use.shape[0], len(list_OD_algorithms)])
    train_scores = np.zeros([X_data_to_use.shape[0], len(list_OD_algorithms)])
    index_arrays = []
    initial_array = np.zeros(len(X_data_to_use), dtype=int)

    for i, clf in enumerate(list_OD_algorithms):
        clf_name = clf.__class__.__name__

        if clf_name == "ABOD":
            X_data_to_use = X_data_to_use.astype(float)

        if clf_name in ["LMDD", "SOD"] and not isinstance(X_data_to_use, np.ndarray):
            X_data_to_use = X_data_to_use.to_numpy()

        clf.fit(X_data_to_use)

        train_scores[:, i] = clf.decision_scores_

        y_pred, y_pred_confidence = clf.predict(X_data_to_use, return_confidence=True)
        y_predproba = clf.predict_proba(X_data_to_use, method="unify", return_confidence=False)

        y_predproba_array[:, i] = y_predproba[:, 1]
        y_pred_confidence_array[:, i] = y_pred_confidence
        index_arrays.append(np.where(y_pred == 1)[0])

        X_data_to_use = X_sc

    for index_array in index_arrays:
        initial_array[index_array] += 1

    return initial_array, y_predproba_array, y_pred_confidence_array, train_scores


def pca_visualization(data, objNames, varNames, save_folder=None, figure_append_name=None):
    model = ho.nipalsPCA(arrX=data, Xstand=False, cvType=["loo"], numComp=10)
    hop.plot(model, comp=[1, 2], plots=[6], objNames=objNames, XvarNames=varNames, save_folder=save_folder, figure_append_name=figure_append_name)

    return model


def scatter_plot_visualization(model, y_predproba_array, y_pred_confidence_array, patient_ids, save_folder=None, figure_append_name=None):
    pca_df = pd.DataFrame(model.X_scores()[:, :2])
    pca_df = pca_df.rename(columns={0: "PC1", 1: "PC2"})
    y_predproba_array_mean = np.median(y_predproba_array, axis=1)
    y_pred_confidence_array_mean = y_pred_confidence_array.mean(axis=1)
    scatter = plt.scatter(
        y_predproba_array_mean,
        y_pred_confidence_array_mean,
        c=np.std(y_predproba_array, axis=1),
        s=np.std(y_pred_confidence_array, axis=1) * 120,
        cmap="viridis",
    )
    plt.xlabel("Median sannsynlighet")
    plt.ylabel("Gjennomsnittlig konfidens")

    for i, txt in enumerate(patient_ids):
        plt.annotate(
            txt,
            (y_predproba_array_mean[i], y_pred_confidence_array_mean[i]),
            fontsize=7,
        )

    cbar = plt.colorbar(scatter)
    cbar.set_label("Standardavvik for sannsynlighet")
    plt.axvline(x=0.8, color="black", linestyle="--", lw=0.8)
    plt.axhline(y=0.9, color="black", linestyle="--", lw=0.8)

    if save_folder:
        if figure_append_name:
            plt.savefig(os.path.join(save_folder, str(figure_append_name+"_"+"Probability VS Confidence.png")), bbox_inches='tight')
        else:
            plt.savefig(os.path.join(save_folder, "Probability VS Confidence.png"), bbox_inches='tight')

    plt.show()

    return pca_df


def pair_plot_visualization(model, y_predproba_array, y_pred_confidence_array, patient_ids, X_data_to_use, save_folder=None, figure_append_name=None):
    data = pd.DataFrame(model.X_scores()[:, :5], columns=["PC1", "PC2", "PC3", "PC4", "PC5"])
    y_predproba_series = pd.Series(np.median(y_predproba_array, axis=1), name="y_predproba")
    data_with_hue = pd.concat([data, y_predproba_series], axis=1)

    g = sns.pairplot(
        data_with_hue,
        hue="y_predproba",
        palette="viridis",
        markers="o",
        plot_kws={"s": 20},
    )

    for i, ax_i in enumerate(g.axes):
        for j, ax_j in enumerate(ax_i):
            if i != j:
                for k, txt in enumerate(patient_ids):
                    if np.median(y_predproba_array, axis=1)[k] > 0.8:
                        ax_j.annotate(txt, (data.iloc[k, j], data.iloc[k, i]), fontsize=7)

    g._legend.remove()
    cax = g.fig.add_axes([0.95, 0.38, 0.015, 0.3])
    sm = plt.cm.ScalarMappable(cmap="viridis")
    sm.set_array(np.median(y_predproba_array, axis=1))
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label("Median sannsynlighet")

    if save_folder:
        if figure_append_name:
            plt.savefig(os.path.join(save_folder, str(figure_append_name + "_" + "Probability Pair plots.png")), bbox_inches='tight')
        else:
            plt.savefig(os.path.join(save_folder, "Probability Pair plots.png"), bbox_inches='tight')

    plt.show()

    data = pd.DataFrame(
        model.X_scores()[:, :5], columns=["PC1", "PC2", "PC3", "PC4", "PC5"]
    )

    y_confidence_series = pd.Series(y_pred_confidence_array.mean(axis=1), name="y_confidence")

    data_with_hue = pd.concat([data, y_confidence_series], axis=1)

    g = sns.pairplot(
        data_with_hue,
        hue="y_confidence",
        palette="viridis",
        markers="o",
        plot_kws={"s": 20},
        diag_kind="kde",
    )

    for i, ax_i in enumerate(g.axes):
        for j, ax_j in enumerate(ax_i):
            if i != j:
                for k, txt in enumerate(patient_ids):
                    if np.median(y_predproba_array, axis=1)[k] > 0.8:
                        ax_j.annotate(
                            txt, (data.iloc[k, j], data.iloc[k, i]), fontsize=7
                        )

    g._legend.remove()

    cax = g.fig.add_axes(
        [0.95, 0.38, 0.015, 0.3]
    )
    sm = plt.cm.ScalarMappable(cmap="viridis")
    sm.set_array(y_pred_confidence_array.mean(axis=1))
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label("Gjennomsnittlig konfidens")

    if save_folder:
        if figure_append_name:
            plt.savefig(os.path.join(save_folder, str(figure_append_name + "_" + "Confidence Pair plots.png")), bbox_inches='tight')
        else:
            plt.savefig(os.path.join(save_folder, "Confidence Pair plots.png"), bbox_inches='tight')

    plt.show()

    data = pd.DataFrame(
        model.X_loadings()[:, :5], columns=["PC1", "PC2", "PC3", "PC4", "PC5"]
    )

    g = sns.pairplot(data)

    for i, ax_i in enumerate(g.axes):
        for j, ax_j in enumerate(ax_i):
            if i != j:
                for k, txt in enumerate(X_data_to_use.columns):
                    ax_j.annotate(txt, (data.iloc[k, j], data.iloc[k, i]), fontsize=7)

    if save_folder:
        if figure_append_name:
            plt.savefig(os.path.join(save_folder, str(figure_append_name + "_" + "Features Pair plots.png")), bbox_inches='tight')
        else:
            plt.savefig(os.path.join(save_folder, "Features Pair plots.png"), bbox_inches='tight')

    plt.show()


def visualiser_OD(X_sc, list_OD_algorithms, patient_ids, visualize, save_folder=None, figure_append_name=None):
    X_data_to_use = X_sc
    data, data_varNames = prepare_data(X_data_to_use)
    data_objNames = patient_ids
    initial_array, y_predproba_array, y_pred_confidence_array, train_scores = detect_outliers(X_data_to_use, X_sc, list_OD_algorithms, patient_ids)

    if visualize:
        if save_folder:
            if figure_append_name:
                model = pca_visualization(data, data_objNames, data_varNames, save_folder, figure_append_name)
                pca_df = scatter_plot_visualization(model, y_predproba_array, y_pred_confidence_array, patient_ids, save_folder, figure_append_name)
                pair_plot_visualization(model, y_predproba_array, y_pred_confidence_array, patient_ids, X_data_to_use, save_folder, figure_append_name)
            else:
                model = pca_visualization(data, data_objNames, data_varNames, save_folder)
                pca_df = scatter_plot_visualization(model, y_predproba_array, y_pred_confidence_array, patient_ids,
                                                    save_folder)
                pair_plot_visualization(model, y_predproba_array, y_pred_confidence_array, patient_ids, X_data_to_use,
                                        save_folder)

        else:
            model = pca_visualization(data, data_objNames, data_varNames)
            pca_df = scatter_plot_visualization(model, y_predproba_array, y_pred_confidence_array, patient_ids)
            pair_plot_visualization(model, y_predproba_array, y_pred_confidence_array, patient_ids, X_data_to_use)


    no_od_df = pd.DataFrame(initial_array, index=patient_ids, columns=["No. OD Detected"])
    return no_od_df, np.median(y_predproba_array, axis=1), y_pred_confidence_array.mean(axis=1), y_predproba_array, y_pred_confidence_array, train_scores
