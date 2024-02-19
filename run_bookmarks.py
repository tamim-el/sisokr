import numpy as np
from Utils.load_data import load_bookmarks
from Methods.SketchedIOKR import IOKR, SIOKR, ISOKR, SISOKR
from Methods.Sketch import SubSample, pSparsified
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import f1_score


# Setting random seed
np.random.seed(seed=42)


# Defining Gaussian kernel
def Gaussian_kernel(gamma):
    def Compute_Gram(X, Y):
        return rbf_kernel(X, Y, gamma=gamma)
    return Compute_Gram


# Loading dataset
X_tr, Y_tr, X_te, Y_te = load_bookmarks()
n_tr = X_tr.shape[0]
n_te = X_te.shape[0]
d = Y_tr.shape[1]
l_bar = np.mean(np.sum(Y_tr, axis=1))


######## SIOKR #######################################################################

print('SIOKR in process...')

# Hyperparameters priorly obtained by inner 5-folds cv
best_L_SIOKR = 1e-09
best_sx_SIOKR = 10000.0
best_sy_SIOKR = l_bar

input_kernel = Gaussian_kernel(gamma=1/(2 * best_sx_SIOKR))
output_kernel = Gaussian_kernel(gamma=1/(2 * best_sy_SIOKR))

# Number of replicates
n_rep = 30

f1_tes = np.zeros(n_rep)
fit_times = np.zeros(n_rep)
decode_times = np.zeros(n_rep)

# Sketch parameters
m = 8000
p = 20.0 / n_tr

rrmse_test_S = np.zeros((n_rep, d))
times_S = np.zeros(n_rep)

for j in range(n_rep):

    R = pSparsified((m, n_tr), p=p, type='Gaussian')

    clf = SIOKR(L=best_L_SIOKR,
                input_kernel=input_kernel,
                output_kernel=output_kernel,
                R=R)

    clf.fit(X_tr, Y_tr)

    Y_pred_te = clf.predict(X_te=X_te)

    f1_tes[j] = f1_score(Y_pred_te, Y_te, average='samples')
    fit_times[j] = clf.fit_time
    decode_times[j] = clf.decode_time

f1_mean = np.mean(f1_tes)
f1_std = 0.5 * np.std(f1_tes)

fit_time_mean = np.mean(fit_times)
fit_time_std = 0.5 * np.std(fit_times)

decode_time_mean = np.mean(decode_times)
decode_time_std = 0.5 * np.std(decode_times)


print('Results obtained with SIOKR on Bookmarks dataset: ')
print('Test F1 score: ' + str(f1_mean) + ' +- ' + str(f1_std))
print('Training time (in seconds): ' + str(fit_time_mean) + ' +- ' + str(fit_time_std))
print('Inference time (in seconds): ' + str(decode_time_mean) + ' +- ' + str(decode_time_std))
print('\n')


######## SISOKR #######################################################################

print('SISOKR in process...')

# Hyperparameters priorly obtained by inner 5-folds cv
best_L_SISOKR = 1e-08
best_sx_SISOKR = 10000.0
best_sy_SISOKR = 10.0

input_kernel = Gaussian_kernel(gamma=1/(2 * best_sx_SISOKR))
output_kernel = Gaussian_kernel(gamma=1/(2 * best_sy_SISOKR))

# Number of replicates
n_rep = 30

f1_tes = np.zeros(n_rep)
fit_times = np.zeros(n_rep)
decode_times = np.zeros(n_rep)

# Sketch parameters
m_in = 8000
m_out = 500
p = 20.0 / n_tr

rrmse_test_S = np.zeros((n_rep, d))
times_S = np.zeros(n_rep)

for j in range(n_rep):

    R_in = SubSample((m_in, n_tr))
    R_out = pSparsified((m_out, n_tr), p=p, type='Gaussian')

    clf = SISOKR(L=best_L_SISOKR,
                input_kernel=input_kernel,
                output_kernel=output_kernel,
                R_in=R_in, R_out=R_out)

    clf.fit(X_tr, Y_tr)

    Y_pred_te = clf.predict(X_te=X_te)

    f1_tes[j] = f1_score(Y_pred_te, Y_te, average='samples')
    fit_times[j] = clf.fit_time
    decode_times[j] = clf.decode_time

f1_mean = np.mean(f1_tes)
f1_std = 0.5 * np.std(f1_tes)

fit_time_mean = np.mean(fit_times)
fit_time_std = 0.5 * np.std(fit_times)

decode_time_mean = np.mean(decode_times)
decode_time_std = 0.5 * np.std(decode_times)


print('Results obtained with SISOKR on Bookmarks dataset: ')
print('Test F1 score: ' + str(f1_mean) + ' +- ' + str(f1_std))
print('Training time (in seconds): ' + str(fit_time_mean) + ' +- ' + str(fit_time_std))
print('Inference time (in seconds): ' + str(decode_time_mean) + ' +- ' + str(decode_time_std))
print('\n')