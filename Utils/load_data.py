import numpy as np
from skmultilearn.dataset import load_from_arff
from scipy.stats import ortho_group
from sklearn.preprocessing import StandardScaler


def load_bibtex(path_tr='Data/bibtex/bibtex-train.arff', path_te='Data/bibtex/bibtex-test.arff', normalize=True):
    """
        Load Dataset Bibtex
    """

    x, y = load_from_arff(path_tr, label_count=159)
    X_tr, Y_tr = x.todense(), y.todense()
    X_tr, Y_tr = np.asarray(X_tr), np.asarray(Y_tr)

    x_test, y_test = load_from_arff(path_te, label_count=159)
    X_te, Y_te = x_test.todense(), y_test.todense()
    X_te, Y_te = np.asarray(X_te), np.asarray(Y_te)

    Y_tr, Y_te = Y_tr.astype(int), Y_te.astype(int)

    # Normalizing
    if normalize:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

    return X_tr, Y_tr, X_te, Y_te


def load_bookmarks(path='Data/bookmarks/bookmarks.arff', n_tr=60000, normalize=True):

    """
        Load Dataset bookmarks
    """

    x, y = load_from_arff(path, label_count=208)
    X, Y = x.todense(), y.todense()

    X_tr, Y_tr = X[:n_tr], Y[:n_tr]
    X_te, Y_te = X[n_tr:], Y[n_tr:]
    
    X_tr = np.asarray(X_tr)
    Y_tr = np.asarray(Y_tr)
    X_te = np.asarray(X_te)
    Y_te = np.asarray(Y_te)

    Y_tr, Y_te = Y_tr.astype(int), Y_te.astype(int)

    # Normalizing
    if normalize:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

    return X_tr, Y_tr, X_te, Y_te


def load_mediamill(path_tr='Data/mediamill/mediamill-train.arff', path_te='Data/mediamill/mediamill-test.arff', normalize=True):

    """
        Load Dataset Mediamill
    """

    x, y = load_from_arff(path_tr, label_count=101)
    X_tr, Y_tr = x.todense(), y.todense()

    x_test, y_test = load_from_arff(path_te, label_count=101)
    X_te, Y_te = x_test.todense(), y_test.todense()
    
    X_tr = np.asarray(X_tr)
    Y_tr = np.asarray(Y_tr)
    X_te = np.asarray(X_te)
    Y_te = np.asarray(Y_te)

    Y_tr, Y_te = Y_tr.astype(int), Y_te.astype(int)

    # Normalizing
    if normalize:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

    return X_tr, Y_tr, X_te, Y_te