# Author: hwanghoseok
# Date  : 2022 - 1 - 26
# RMSVM solver

#%%
from numpy.core.fromnumeric import shape, size
import pandas as pd
import numpy as np
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
import random as rnd

#%%

# generate simplex W = {W1, W2, ...}

def XI_gen(k): # k-1 * k
    tempA = -(1.0 + np.sqrt(k))/ ((k - 1)**(1.5))
    tempB = tempA + np.sqrt(k/(k - 1))
    
    XI = np.full((k-1, k), tempA) # tempA 다 더해진 k-1 * k 행렬 생성
    XI[:,0] = 1.0 / np.sqrt(k / (k - 1)) 
    
    for i in range(k-1):
        XI[i, i+1] = tempB
    return(XI)



# y matrix generate

def Y_matrix_gen(k, nobs, y):
    XI = XI_gen(k = k)
    
    Y_matrix = np.zeros((nobs, k-1)) # n * k-1 
    
    for i in range(nobs-1):
        Y_matrix[i,:] = XI[:,y[i]]
    return(Y_matrix)
    

# ramsvm code
y = [0,1,2,1]

def code_ramsvm(y):
    n_class = len(np.unique(y)) # class number
    n = len(y) # count
    yyi = Y_matrix_gen(k = n_class, nobs = n, y = y)
    W = XI_gen(n_class)
    
    y_index = np.column_stack((np.arange(n), y))
    index_mat = np.full((n, n_class), -1) # n * k 
    index_mat[y_index] = 1

    Hmatq = list()
    Lmatq = list()
    
    for q in range(n_class - 1):
        Hmatq_temp = 'Null'
        Lmatq_temp = 'NULL'
        for i in range(n_class):
            temp = np.eye(n) * W[q, i]
            Ltemp = np.eye(temp) * index_mat[:,i]
            Hmatq_temp = np.vstack((Hmatq_temp, temp))
            Lmatq_temp = np.hstack((Lmatq_temp, Ltemp))
        
        Hmatq[q] = Hmatq_temp
        Lmatq[q] = Lmatq_temp
    
    return(list(yyi = yyi, W = W, Hmatq = Hmatq, Lmatq = Lmatq, y_index = y_index))
            

y_index = np.column_stack((np.arange(4), y)) # 4개 데이터 클래스 3개
index_mat = np.full((4, 3), -1) # n * k 

index_mat[y_index,:] = 1

