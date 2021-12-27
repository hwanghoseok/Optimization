import numpy as np
import matplotlib.pyplot as plt
#%%

alpha = 0.1
beta = 0.7
mu = 5

n = 20
d = 15

eps = 10e-6

lambd = 10

x = np.random.rand(n,d) # 20 * 15
p = np.random.rand(n,d) # 20 * 15
Q = np.eye(n) # 20 * 20
x0 = np.zeros((n,d)) # 20 * 15

p.shape

def function_true(x, Q, p):
    """ function = 1/2*xTQx + pTx
    st. 1/2*xTQ[i]x + pT[i]x <= 0, i = 1, ... ,m
    """
    return 1/2 * np.dot(x.T, np.dot(Q, x)) + np.dot(p.T, x)

function_true(x, Q, p).shape

def function(x, Q, p, t0):
    """ function = t(1/2*xTQx + pTx) + phi(x)
    Q, p : 쿼드라틱 폼의 메트릭스
    """
    return - sum([np.log(- 1/2 * np.dot(x, np.dot(Q[i], x)) - np.dot(p[i], x)) for i in range(Q.shape[0])])

np.dot(x, np.dot(Q[0], x)) - np.dot(p[0], x)

Q[0].shape
np.dot(x,np.dot(Q[0],x)).shape