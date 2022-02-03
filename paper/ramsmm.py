# Author: hwanghoseok
# Date  : 2022 - 1 - 26
# RMSVM solver

#%%
from numpy.core.fromnumeric import shape, size
from numpy.lib.arraysetops import unique
import pandas as pd
import numpy as np
from numpy.linalg import svd

#%%

# generate simplex W = {W1, W2, ...}

def XI_gen(k): # k-1 * k
    tempA = -(1.0 + np.sqrt(k))/ ((k - 1)**(1.5))
    tempB = tempA + np.sqrt(k/(k - 1))
    
    XI = np.full((k-1, k), tempA) # tempA 다 더해진 k-1 * k 행렬 생성
    XI[:,0] = 1.0 / np.sqrt(k - 1) 
    
    for i in range(k-1):
        XI[i, i+1] = tempB
    return(XI)


# y matrix generate

def Y_matrix_gen(k, nobs, y):
    XI = XI_gen(k = k)
    
    Y_matrix = np.zeros((nobs, k-1)) # n * k-1 
    
    for i in range(nobs):
        Y_matrix[i,:] = XI[:,y[i]-1]
    return(Y_matrix)


# ramsvm code

def code_ramsvm(y):
    n_class = len(np.unique(y)) # class number
    n = len(y) # count
    yyi = Y_matrix_gen(k = n_class, nobs = n, y = y)
    W = XI_gen(n_class)
    
    y_index = np.column_stack((np.arange(n), y))
    index_mat = np.full((n, n_class), -1) # n * k 
    for i in range(n):
        index_mat[i,(y_index[i,1]-1)] = 1

    Hmatq = np.empty((n*n_class, n, n_class - 1)) # nk * n * q
    Lmatq = np.empty((n*n_class, n_class - 1)) # nk * q
    
    for q in range(n_class - 1):
        Hmatq_temp = np.zeros(shape = (0,n))
        Lmatq_temp = np.zeros(shape = (n,0))
        for j in range(n_class):
            temp = np.eye(n) * W[q, j]
            temp = temp * index_mat[:,j]
            Hmatq_temp = np.append(Hmatq_temp, temp, axis = 0)
            Lmatq_temp = np.append(Lmatq_temp, np.diag(temp))
        Hmatq[:,:,q] = Hmatq_temp
        Lmatq[:,q] = Lmatq_temp
    
    return yyi, W, Hmatq, Lmatq, y_index

# Singular value thesholding Dt, 

def Shirinkage(X, tau): 
    U, Sigma, Vt = svd(X) # p*p, min(p,q) * min(p*q), q*q
    sig = np.maximum(0, Sigma - tau) 
    
    d = np.zeros(shape=(len(U),len(Vt)))
    
    for i in range(min(len(U),len(Vt))):
        d[i,i] = sig[i]
    D = U @ d @ Vt
    
    return D

# SMM core notation

def QPnot(X, y, rho, lamb = 0.5, wedge, S): 
    """
    Args:
        X (tensor): [p * q * (n_class - 1)]
        y (list): [n(data count)]
        rho (list): [n_class - 1]
        wedge (tensor): [p * q * (n_class - 1)]
        S (tensor): [p * q * (n_class - 1)]
        lamb (float, optional): Defaults.
    """
    n = len(y)
    n_class = len(np.unique(y))
    
    Q = np.empty((n, n, n_class - 1)) # n * n * k-1
    p = np.empty((n, n_class-1)) # n * k-1
    
    for q in range(n_class):
        for i in range(n):
            Xi = np.trace((wedge[:,:,q] + rho[q] * S[:,:,q]).T.dot(X[:,:,i]))
            p[i,q] = Xi/ (rho[q] + n * lamb)   
            for j in range(n):
                Xij = np.trace(X[:,:,i].T.dot(X[:,:,j]))
            Q[i,j,q] = Xij / (rho[q] + n * lamb)
    return Q, p


def ramsvm_compact(Q, y, gamma = 0.5, lambda, epsilon = 1.0e-6, eig_tol_d = 0, epsilon_d =1.0e-6):
    
    n_class = len(unique(y))
    n = len(y)
    
    yyi, W, Hmatq, Lmatq, y_index = code_ramsvm(y)
    
    Q_til = np.empty((n*n_class, n*n_class))
    for q in range(n_class):
        Q_til += np.dot(np.dot(Hmatq[:,:,q],Q[:,:,q]),Hmatq[:,:,q].T)
        p_til1 += np.dot(Hmatq[:,:,q],p[:,q])
    return Q_til, p_til

    

