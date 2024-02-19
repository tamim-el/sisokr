import numpy as np
from time import time


class IOKR:
    
    def __init__(self, L, input_kernel, output_kernel, verbose=0):
        self.X_tr = None
        self.Y_tr = None
        self.L = L
        self.input_kernel = input_kernel
        self.output_kernel = output_kernel
        self.M = None
        self.fit_time = None
        self.decode_time = None
        self.verbose = verbose
        
    def fit(self, X, Y):
        
        t0 = time()
        self.X_tr = X.copy()
        self.Y_tr = Y.copy()
        Kx = self.input_kernel(self.X_tr, Y=self.X_tr)
        n = Kx.shape[0]
        self.M = np.linalg.inv(Kx + n * self.L * np.eye(n))
        self.fit_time = time() - t0
        if self.verbose > 0:
            print(f'Fitting time: {self.fit_time}')

    def predict(self, X_te, Y_c=None):
        
        t0 = time()
        if Y_c is None:
            _, indices = np.unique(self.Y_tr, axis=0, return_index=True)
            Y_c = self.Y_tr[np.sort(indices)].copy()
        Kx = self.input_kernel(X_te, Y=self.X_tr)
        Ky = self.output_kernel(self.Y_tr, Y=Y_c)
        scores = Kx.dot(self.M).dot(Ky)
        idx_pred = np.argmax(scores, axis=1)
        self.decode_time = time() - t0
        if self.verbose > 0:
            print(f'Decoding time: {self.decode_time}')
        
        return Y_c[idx_pred]

    def predict_linear(self, X_te):

        K_x_te_tr = self.input_kernel(X_te, self.X_tr)
        Y_pred = K_x_te_tr.dot(self.M).dot(self.Y_tr)

        return Y_pred

    def mse(self, X_te, Y_test, verbose=0):

        K_x_tr_te = self.input_kernel(self.X_tr, X_te)
        n_te = K_x_tr_te.shape[1]
        A = K_x_tr_te.T.dot(self.M)
        K_y = self.output_kernel(self.Y_tr, self.Y_tr)

        K_y_tr_te = self.output_kernel(self.Y_tr, Y_test)
        norm_h = np.diag(A.dot(K_y).dot(A.T))
        prod_h_y = np.diag(A.dot(K_y_tr_te))
        K_y_te_te = self.output_kernel(Y_test, Y_test)
        norm_y = np.diag(K_y_te_te)

        se = norm_h - 2 * prod_h_y + norm_y
        mse = np.mean(se)
        std = np.std(se) / np.sqrt(n_te)

        if verbose == 1:
            print(np.sqrt(norm_h[:5]), prod_h_y[:5], np.sqrt(norm_y[:5]), flush=True)
            print(se[:5], flush=True)
            print(np.mean(norm_h), np.mean(prod_h_y), np.mean(norm_y), flush=True)

        return mse, std



class SIOKR:
    
    def __init__(self, L, input_kernel, output_kernel, R, mu=1e-8, verbose=0):
        self.X_tr = None
        self.Y_tr = None
        self.L = L
        self.input_kernel = input_kernel
        self.output_kernel = output_kernel
        self.R = R
        self.mu = mu
        self.M = None
        self.fit_time = None
        self.decode_time = None
        self.verbose = verbose
        
    def fit(self, X, Y):
        
        t0 = time()
        self.X_tr = X.copy()
        self.Y_tr = Y.copy()   
        n = X.shape[0]
        m = self.R.size[0]
        self.Y_tr = Y
        KRT = self.R.multiply_Gram_one_side(X, self.input_kernel, X)
        RKRT = self.R.multiply_matrix_one_side(KRT, right=False)
        B = KRT.T.dot(KRT) + n * self.L * RKRT
        B_inv = np.linalg.inv(B + self.mu * np.eye(m))
        self.M = B_inv.dot(KRT.T)
        self.fit_time = time() - t0
        if self.verbose > 0:
            print(f'Fitting time: {self.fit_time}')

    def predict(self, X_te, Y_c=None):
        
        t0 = time()
        if Y_c is None:
            _, indices = np.unique(self.Y_tr, axis=0, return_index=True)
            Y_c = self.Y_tr[np.sort(indices)].copy()
        K_te_trRT = self.R.multiply_Gram_one_side(X_te, self.input_kernel, Y=self.X_tr)
        Ky = self.output_kernel(self.Y_tr, Y=Y_c)
        scores = (K_te_trRT.dot(self.M)).dot(Ky)
        idx_pred = np.argmax(scores, axis=1)
        self.decode_time = time() - t0
        if self.verbose > 0:
            print(f'Decoding time: {self.decode_time}')
        
        return Y_c[idx_pred]

    def predict_linear(self, X_te):

        K_x_te_trRT = self.R.multiply_Gram_one_side(X_te, self.input_kernel, Y=self.X_tr)
        Y_pred = K_x_te_trRT.dot(self.M).dot(self.Y_tr)

        return Y_pred

    def mse(self, X_te, Y_test, verbose=0):

        K_x_te_trRT = self.R.multiply_Gram_one_side(X_te, self.input_kernel, Y=self.X_tr)
        n_te = K_x_te_trRT.shape[0]
        A = K_x_te_trRT.dot(self.M)
        K_y = self.output_kernel(self.Y_tr, self.Y_tr)

        K_y_tr_te = self.output_kernel(self.Y_tr, Y_test)
        norm_h = np.diag(A.dot(K_y).dot(A.T))
        prod_h_y = np.diag(A.dot(K_y_tr_te))
        K_y_te_te = self.output_kernel(Y_test, Y_test)
        norm_y = np.diag(K_y_te_te)

        se = norm_h - 2 * prod_h_y + norm_y
        mse = np.mean(se)
        std = np.std(se) / np.sqrt(n_te)

        if verbose == 1:
            print(np.sqrt(norm_h[:5]), prod_h_y[:5], np.sqrt(norm_y[:5]), flush=True)
            print(se[:5], flush=True)
            print(np.mean(norm_h), np.mean(prod_h_y), np.mean(norm_y), flush=True)

        return mse, std



class ISOKR:
    
    def __init__(self, L, input_kernel, output_kernel, R, mu=0, verbose=0):
        self.X_tr = None
        self.Y_tr = None
        self.L = L
        self.input_kernel = input_kernel
        self.output_kernel = output_kernel
        self.R = R
        self.mu = mu
        self.KyRT = None
        self.RKyRT = None
        self.M = None
        self.fit_time = None
        self.decode_time = None
        self.verbose = verbose
        
    def fit(self, X, Y):
        
        t0 = time()
        self.X_tr = X.copy()
        self.Y_tr = Y.copy()
        Kx = self.input_kernel(X, X)
        n = Kx.shape[0]
        m = self.R.size[0]
        Omega = np.linalg.inv(Kx + n * self.L * np.eye(n))
        self.KyRT = self.R.multiply_Gram_one_side(Y, self.output_kernel, Y)
        self.RKyRT = self.R.multiply_matrix_one_side(self.KyRT, right=False)
        RKyRT_inv = np.linalg.inv(self.RKyRT.copy() + self.mu * np.eye(m))
        self.M = Omega.dot(self.KyRT).dot(RKyRT_inv)
        self.fit_time = time() - t0
        if self.verbose > 0:
            print(f'Fitting time: {self.fit_time}')

    def predict(self, X_te, Y_c=None):
        
        t0 = time()
        if Y_c is None:
            _, indices = np.unique(self.Y_tr, axis=0, return_index=True)
            Y_c = self.Y_tr[np.sort(indices)].copy()
            _, indices = np.unique(self.KyRT.T, axis=1, return_index=True)
            RKy = self.KyRT.T[:, np.sort(indices)].copy()
        else:
            RKy = self.R.multiply_Gram_one_side(self.Y_tr, self.output_kernel, Y_c, right=False)
        Kx = self.input_kernel(X_te, self.X_tr)
        scores = Kx.dot(self.M).dot(RKy)
        idx_pred = np.argmax(scores, axis=1)
        self.decode_time = time() - t0
        if self.verbose > 0:
            print(f'Decoding time: {self.decode_time}')
        
        return Y_c[idx_pred]

    def predict_linear(self, X_te):

        K_x_te_tr = self.input_kernel(X_te, self.X_tr)
        RY = self.R.multiply_matrix_one_side(self.Y_tr, right=False)
        Y_pred = K_x_te_tr.dot(self.M).dot(RY)

        return Y_pred

    def mse(self, X_te, Y_test, verbose=0):

        K_x_tr_te = self.input_kernel(self.X_tr, X_te)
        n_te = K_x_tr_te.shape[1]
        A = K_x_tr_te.T.dot(self.M)

        RKy_tr_te = self.R.multiply_Gram_one_side(self.Y_tr, self.output_kernel, Y_test, right=False)
        norm_h = np.diag(A.dot(self.RKyRT).dot(A.T))
        prod_h_y = np.diag(A.dot(RKy_tr_te))
        K_y_te_te = self.output_kernel(Y_test, Y_test)
        norm_y = np.diag(K_y_te_te)

        se = norm_h - 2 * prod_h_y + norm_y
        mse = np.mean(se)
        std = np.std(se) / np.sqrt(n_te)

        if verbose == 1:
            print(np.sqrt(norm_h[:5]), prod_h_y[:5], np.sqrt(norm_y[:5]), flush=True)
            print(se[:5], flush=True)
            print(np.mean(norm_h), np.mean(prod_h_y), np.mean(norm_y), flush=True)

        return mse, std



class SISOKR:
    
    def __init__(self, L, input_kernel, output_kernel, R_in, R_out, mu_in=1e-8, mu_out=0, verbose=0):
        self.X_tr = None
        self.Y_tr = None
        self.L = L
        self.input_kernel = input_kernel
        self.output_kernel = output_kernel
        self.R_in = R_in
        self.R_out = R_out
        self.mu_in = mu_in
        self.mu_out = mu_out
        self.KyRT = None
        self.RKyRT = None
        self.M = None
        self.fit_time = None
        self.decode_time = None
        self.verbose = verbose
        
    def fit(self, X, Y):
        
        t0 = time()
        self.X_tr = X.copy()
        self.Y_tr = Y.copy()
        self.input_kernel = self.input_kernel
        self.output_kernel = self.output_kernel
        n = X.shape[0]
        m_in = self.R_in.size[0]
        m_out = self.R_out.size[0]
        KRT = self.R_in.multiply_Gram_one_side(X, self.input_kernel, X)
        RKRT = self.R_in.multiply_matrix_one_side(KRT, right=False)
        B = KRT.T.dot(KRT) + n * self.L * RKRT
        Omega = np.linalg.inv(B + self.mu_in * np.eye(m_in)).dot(KRT.T)
        self.KyRT = self.R_out.multiply_Gram_one_side(Y, self.output_kernel, Y)
        self.RKyRT = self.R_out.multiply_matrix_one_side(self.KyRT, right=False)
        RKyRT_inv = np.linalg.inv(self.RKyRT.copy() + self.mu_out * np.eye(m_out))
        self.M = Omega.dot(self.KyRT).dot(RKyRT_inv)
        self.fit_time = time() - t0
        if self.verbose > 0:
            print(f'Fitting time: {self.fit_time}')

    def predict(self, X_te, Y_c=None):
        
        t0 = time()
        if Y_c is None:
            _, indices = np.unique(self.Y_tr, axis=0, return_index=True)
            Y_c = self.Y_tr[np.sort(indices)].copy()
            _, indices = np.unique(self.KyRT.T, axis=1, return_index=True)
            RKy = self.KyRT.T[:, np.sort(indices)].copy()
        else:
            RKy = self.R_out.multiply_Gram_one_side(self.Y_tr, self.output_kernel, Y_c, right=False)
        KxRT = self.R_in.multiply_Gram_one_side(X_te, self.input_kernel, Y=self.X_tr)
        scores = KxRT.dot(self.M).dot(RKy)
        idx_pred = np.argmax(scores, axis=1)
        self.decode_time = time() - t0
        if self.verbose > 0:
            print(f'Decoding time: {self.decode_time}')
        
        return Y_c[idx_pred]

    def predict_linear(self, X_te):

        K_x_te_trRT = self.R_in.multiply_Gram_one_side(X_te, self.input_kernel, Y=self.X_tr)
        RY = self.R_out.multiply_matrix_one_side(self.Y_tr, right=False)
        Y_pred = K_x_te_trRT.dot(self.M).dot(RY)

        return Y_pred

    def mse(self, X_te, Y_test, verbose=0):

        K_x_te_trRT = self.R_in.multiply_Gram_one_side(X_te, self.input_kernel, Y=self.X_tr)
        n_te = K_x_te_trRT.shape[0]
        A = K_x_te_trRT.dot(self.M)

        RKy_tr_te = self.R_out.multiply_Gram_one_side(self.Y_tr, self.output_kernel, Y_test, right=False)
        norm_h = np.diag(A.dot(self.RKyRT).dot(A.T))
        prod_h_y = np.diag(A.dot(RKy_tr_te))
        K_y_te_te = self.output_kernel(Y_test, Y_test)
        norm_y = np.diag(K_y_te_te)

        se = norm_h - 2 * prod_h_y + norm_y
        mse = np.mean(se)
        std = np.std(se) / np.sqrt(n_te)

        if verbose == 1:
            print(np.sqrt(norm_h[:5]), prod_h_y[:5], np.sqrt(norm_y[:5]), flush=True)
            print(se[:5], flush=True)
            print(np.mean(norm_h), np.mean(prod_h_y), np.mean(norm_y), flush=True)

        return mse, std