import pandas as pd
import numpy as np
from scipy.linalg import sqrtm


# Whitening
#   There are two things we are trying to accomplish with whitening:
#   Make the features less correlated with one another.
#   Give all of the features the same variance.
# Whitening has two simple steps:
#   Project the dataset onto the eigenvectors.
#   This rotates the dataset so that there is no correlation between the components.
#   Normalize the the dataset to have a variance of 1 for all components.
#   This is done by simply dividing each component by the square root of its eigenvalue.

def check_symmetric(a, tol=1e-8):
    return np.allclose(a, a.T, atol=tol)


def compute_cov(x: np.ndarray, mu_s):
    assert len(x.shape) == 1
    return np.dot((x - mu_s).reshape(-1, 1), (x - mu_s).reshape(1, -1))


def spectral_mirror(X: pd.DataFrame, y: pd.Series, i):
    assert len(np.unique(y)) == 2  # binary classifier only
    # need to have n>d samples in order to accurately reconstruct the sub-space
    # todo: what 'i' stands for?
    d = X.shape[1]
    n = X.shape[0]
    assert len(X) == len(y)
    half = int(len(X) / 2)
    mu_s = np.average(X[:half, :], axis=0)
    assert mu_s.shape == (d,)
    sum_covs = 0
    for i in range(half):  # how to vectorize this # line2
        sum_covs += compute_cov(X[i], mu_s)
    sig_s = sum_covs / half
    assert sig_s.shape == (d, d)
    sig_s_inverse = np.linalg.inv(sig_s)
    sum_rs = 0
    for i in range(half):  # how to vectorize this # line3
        sum_rs += y[i] * np.dot(sig_s_inverse, (X[i] - mu_s))
    r_s = sum_rs / half
    assert r_s.shape == (d,)
    print("pre-processing complete")
    top_half_count = n - half + 1
    Z_s = np.zeros((top_half_count,))
    for i in range(half, n):  # line 4
        Z_s[i - half] = y[i] * np.sign(np.dot(r_s, X[i]))
    # line 5
    sum_Q_s = 0
    square_root_inverse_cov = sqrtm(sig_s_inverse)
    for i in range(half, n):  # line 5
        sum_Q_s += Z_s[i - half] * np.dot(np.dot(square_root_inverse_cov, compute_cov(X[i], mu_s)),
                                          square_root_inverse_cov)
    Q_s = sum_Q_s / top_half_count
    assert check_symmetric(Q_s)
    print(Q_s.shape)
    print("svd")
    # the eigen_vectors structure of Q enables to estimate the span of u1 to uk, which are the parameters for the classifiers.
    eigen_values, eigen_vectors = np.linalg.eig(Q_s)  # line 6

    eigen_values_median = np.median(eigen_values)
    out_eignvectors = []
    for idx, eigen_value in enumerate(eigen_values):  # line 7
        if eigen_value > eigen_values_median:
            out_eignvectors.append(eigen_vectors[idx])

    return out_eignvectors

class LogisticRegression:
    def __init__(self, w: np.ndarray):
        self.w = w

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def predict(self, X: np.ndarray):
        preds = np.zeros((X.shape[0]))
        for i in range(len(preds)):
            preds[i] = self.sigmoid(np.dot(self.w, X[i, :]))
        return np.round(preds)


class MixtureLogisticRegression:
    def __init__(self, vectors):
        self.d = vectors[0].shape[0]
        self.n_classifiers = len(vectors)
        self.classifiers = []
        for idx in range(self.n_classifiers):
            classifier = LogisticRegression(vectors[idx])
            self.classifiers.append(classifier)


    def predict(self, X: np.ndarray):
        assert len(X.shape) == 2
        assert X.shape[1] == self.d  # trained on the same data
        results = np.zeros((X.shape[0], self.n_classifiers)).astype(int)
        for i in range(self.n_classifiers):
            results[:, i] = self.classifiers[i].predict(X)
        counts = np.apply_along_axis(np.bincount, axis=1, arr=results)
        return np.argmax(counts, axis= 1)

from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

if __name__ == '__main__':
    data_c = load_breast_cancer()
    X = data_c.data[:, :]  # take only 4 features
    y = data_c.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    k_eigen_vectors = spectral_mirror(X_train, y_train, 30)
    print("calculated k_eigen_vectors")

    mixture = MixtureLogisticRegression(k_eigen_vectors,)
    score = accuracy_score(y_test, mixture.predict(X_test))

    print("labels: dist [0 1]:", np.bincount(y_test))
    print("labels: examples:", len(y_test), "d:", X_test.shape[1])
    print('accuracy_score: spectral_mirror:', score)

    ref_naivebayes = GaussianNB()
    ref_naivebayes.fit(X_train, y_train)

    print('accuracy_score: GaussianNB:', accuracy_score(y_test, ref_naivebayes.predict(X_test)))





