from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import check_array

from pyod_zryan.base import BaseDetector

class Ensemble(BaseDetector):

    def __init__(self, contamination=0.1, n_jobs=1, patient_ids=None, x_columns=None, initialized_modules=None, clfs=[]):
        super(Ensemble, self).__init__(contamination=contamination)
        self.n_jobs = n_jobs
        self.patients_ids = patient_ids
        self.x_columns = x_columns
        self.initialized_modules = initialized_modules
        self.clfs = clfs

    def fit(self, X, y=None):
        """Fit detector. y is ignored in unsupervised methods.
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.
        y : Ignored
            Not used, present for API consistency by convention.
        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = check_array(X)
        self._set_n_classes(y)
        self.decision_scores_ = self.decision_function(X)
        self.X_train = X
        self._process_decision_scores()

        for i, clf in enumerate(self.initialized_modules):
            clf_name = clf.__class__.__name__

            if clf_name == "ABOD":
                X = X.astype(float)
            if clf_name in ["LMDD", "SOD"] and not isinstance(X, np.ndarray):
                X = X.to_numpy()

            self.clfs.append(clf.fit(X))

        return self

    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.
         For consistency, outliers are assigned with larger anomaly scores.
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.
        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        original_size = X.shape[0]

        y_predproba_array = np.zeros([X.shape[0], len(self.initialized_modules)])

        for i, clf in enumerate(self.clfs):

            y_predproba = clf.predict_proba(X, method="unify", return_confidence=False)

            y_predproba_array[:, i] = y_predproba[:, 1]

        # Compute the median for each row
        median_values = np.median(y_predproba_array, axis=1)

        decision_scores_ = median_values

        decision_scores_ = decision_scores_[-original_size:]

        return decision_scores_.ravel()


    def explain_outlier(self, ind, columns=None, cutoffs=None,
                        feature_names=None, file_name=None,
                        file_type=None):  # pragma: no cover
        """Plot dimensional outlier graph for a given data point within
        the dataset.

        Parameters
        ----------
        ind : int
            The index of the data point one wishes to obtain
            a dimensional outlier graph for.

        columns : list
            Specify a list of features/dimensions for plotting. If not
            specified, use all features.

        cutoffs : list of floats in (0., 1), optional (default=[0.95, 0.99])
            The significance cutoff bands of the dimensional outlier graph.

        feature_names : list of strings
            The display names of all columns of the dataset,
            to show on the x-axis of the plot.

        file_name : string
            The name to save the figure

        file_type : string
            The file type to save the figure

        Returns
        -------
        Plot : matplotlib plot
            The dimensional outlier graph for data point with index ind.
        """
        if columns is None:
            columns = list(range(self.O.shape[1]))
            column_range = range(1, self.O.shape[1] + 1)
        else:
            column_range = range(1, len(columns) + 1)

        cutoffs = [1 - self.contamination,
                   0.99] if cutoffs is None else cutoffs

        # plot outlier scores
        plt.scatter(column_range, self.O[ind, columns], marker='^', c='black',
                    label='Outlier Score')

        for i in cutoffs:
            plt.plot(column_range,
                     np.quantile(self.O[:, columns], q=i, axis=0),
                     '--',
                     label='{percentile} Cutoff Band'.format(percentile=i))
        plt.xlim([1, max(column_range)])
        plt.ylim([0, int(self.O[:, columns].max().max()) + 1])
        plt.ylabel('Dimensional Outlier Score')
        plt.xlabel('Dimension')

        ticks = list(column_range)
        if feature_names is not None:
            assert len(feature_names) == len(ticks), \
                "Length of feature_names does not match dataset dimensions."
            plt.xticks(ticks, labels=feature_names)
        else:
            plt.xticks(ticks)

        plt.yticks(range(0, int(self.O[:, columns].max().max()) + 1))
        plt.xlim(0.95, ticks[-1] + 0.05)
        label = 'Outlier' if self.labels_[ind] == 1 else 'Inlier'
        plt.title(
            'Outlier score breakdown for sample #{index} ({label})'.format(
                index=ind + 1, label=label))
        plt.legend()
        plt.tight_layout()

        if file_name is not None:
            if file_type is not None:
                plt.savefig(file_name + '.' + file_type, dpi=300)
            else:
                plt.savefig(file_name + '.' + 'png', dpi=300)
        plt.show()
