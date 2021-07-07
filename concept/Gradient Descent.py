###### 경사하강법(gradien descent)
# 최적 모델 파라미터 찾기, 비용함수의 그래디언트가 0이 되는 방향으로 업데이트
# 최적 추정치 1.수식으로 계산 어렵거나 2.변수가 너무 많거나 3.training set이 클 경우 사용
# 즉, 경사 하강법은 임의 초기값에서 시작하여 비용함수가 감소하는 방향으로 파라미터 업데이트해 최솟값에 수렴하도록 함.

# 그래디언트는 비용함수의 미분한 값, 파라미터가 여러개일 땐 편미분
# 비용함수는 기준으로 정의하기 나름 ex) mse



####### 1.배치 경사 하강법 BGD

# 가장 기본, 파라미터 2개 이상인 경우
# 매 스텝에서 training set 전체를 사용해 그래디언트 계산 및 파라미터 업데이트


import numpy as np
import matplotlib.pyplot as plt

# y = b0 + b1*x + e, true_B = [6, -2]

X = 1.5 * np.random.rand(100,1) # 0 ~ 1 균일분포, 표준 정규분포 난수를 matrix array(a, b) 생성
y = 6 - 2 * np.random.randn(100, 1) # 평균 0 표준편차 1의 가우시안, 표준 정규분포 난수를 matrix array(a, b) 생성
# true y

# np.c_ : 두개의 1차원 배열을 칼럼으로 세로로 붙여서 2차원 배열 만듦
# np.ones : 1 로 가득찬 array 생성
X_b = np.c_[np.ones((100, 1)), X]  # 100 * 2


# 배치 경사법으로 B추정치 B_hat 구하기

eta = 0.01 # 내려가는 스텝의 크기
n_iter = 10000 # 10000번 반복
m = 100 # sample 수
B_hat_BGD = np.random.randn(2,1) # B_hat 초기값, 2 * 1

for iteration in range(n_iter):
    # 2 * 1 = 2/m * (2 * 100) * ((100 * 2)(2 * 1) - (100 * 1)), 오차제곱합을 미분한 식
    gradients = 2/m * X_b.T.dot(X_b.dot(B_hat_BGD) - y) 
    B_hat_BGD = B_hat_BGD - eta * gradients

B_hat_BGD # true value [6, -2]와 유사

# [이론 정리]
# mce는 contrast function을 최소화 하는 추정치, contrast ft에 따라 추정치 명칭 달라짐
# mle : contrast function으로 likelihood 사용
# lse : contrast function으로 오차제곱합 사용

# 선형 회귀에서 정규성 가정시 정규분포 maximum likelihood가 오차제곱합 식과 비슷해짐
# 따라서 b hat 구할 때 mle = lse
# 선형 회귀에서 오차의 정규분포 가정시 정규분포의 표준편차 추정치 sigma hat을 구할 땐 mle ^= lse

# B의 LSE(최소제곱추정치)와 B_hat_BGD 비교, LSE = (t(x)*x)^(-1) * t(x) * y
B_hat = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)  # linalg.inv : 역행렬

# 학습률 0, 0.01, 0.001일 때 각각 파라미터가 수렴해가는 과정 확인
B_path_BGD = []

def plot_gradient_descent(B_hat, eta, B_path = None):
    m = 100 # sample 수
    plt.plot(X, y, "b.")
    n_iter = 10000 # 반복수
    for iter in range(n_iter):
        if iter % 1000 == 0:
            X_new = np.array([[0], [1.5]]) # x의 범위 [0, 1.5]
            X_new_b = np.c_[np.ones((2,1)), X_new]
            y_predict = X_new_b.dot(B_hat)
            
            style = "r-" if iter > 0 else "g--"
            plt.plot(X_new, y_predict, style)
        gradients = 2/m * X_b.T.dot(X_b.dot(B_hat) - y)
        B_hat = B_hat - eta * gradients
        if B_path is not None:
            B_path.append(B_hat)
    plt.xlabel("$x_1", fontsize = 17)
    plt.axis([0, 1.5, 0, 10])
    plt.title(r"$\eta = {}$". format(eta), fontsize = 16)

np.random.seed(42)

B_hat_BGD = 2*np.random.randn(2,1)  # random initialization

plt.figure(figsize=(10,4))
plt.subplot(131); plot_gradient_descent(B_hat_BGD, eta=0.0)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.subplot(132); plot_gradient_descent(B_hat_BGD, eta=0.01, B_path=B_path_BGD)
plt.subplot(133); plot_gradient_descent(B_hat_BGD, eta=0.001, B_path=B_path_BGD)

plt.show()

# 하이퍼 파라미터 조정해서 왜 lse 아니라 경사하강법 사용?
# 변수개수가 증가 할 수록 lse 계산 복잡도 증가하기 때문





###### 2.확률적 경사 하강법 stochastic gradient descent

# 장점 : 배치 경사 하강법에 비해 빠름, 하나의 샘플에 대한 그래디언트만 계산, 데이터가 매우 클 때 유용함
# 장점 : 배치 경사 하강법에 비해 비용함수가 불규칙할 경우 알고리즘이 지역 최솟값을 건너뛸 가능성이 높음

# 단점 : 배치 경사 하강법에 비해 불안정, 하이퍼 파라미터 증가
# 단점 : 비용함수가 일정하게 감소하지 않고 요동치기 때문에 평균적으로 배치 경사 하강법에 비해 전역 최솟값에 다다르지 못함

# 해결 : 학습 스케쥴(leaning schedule) 설정, 처음에 학습 크게해 지역 최솟값에 빠지지 않도록 한 후 점차 줄여 전역 최솟값에 도달.

# 학습스케쥴 적용 x
n_epochs = 50
eta = 0.01

B_hat_SGD = np.random.randn(2, 1) # 초기화

for epoch in range(n_epochs):
    for i in range(m):
        stoch_index = np.random.randint(m) # 0 ~ m 사이의 랜덤 정수(int) 반환
        xi = X_b[stoch_index:stoch_index+1]
        yi = y[stoch_index:stoch_index+1]
        
        gradients = 2 * xi.T.dot(xi.dot(B_hat_SGD) - yi)
        B_hat_SGD = B_hat_SGD - eta * gradients

B_hat_SGD


# 학습스케줄 적용 o
n_epochs = 50

t0, t1 = 5, 50 # 학습 스케쥴 하이퍼 파라미터
def learning_schedule(t):
    return t0 / (t + t1)

B_hat_SGD = np.random.randn(2, 1) # 초기화

t = 0
for epoch in range(n_epochs):
    for i in range(m):
        t += 1
        stoch_index = np.random.randint(m) # 0 ~ m 사이의 랜덤 정수(int) 반환
        xi = X_b[stoch_index:stoch_index+1]
        yi = y[stoch_index:stoch_index+1]
        eta = learning_schedule(t)
        gradients = 2 * xi.T.dot(xi.dot(B_hat_SGD) - yi)
        B_hat_SGD = B_hat_SGD - eta * gradients

B_hat_SGD




###### 3.미니 배치 경사 하강법

# 확률적 경사 하강법과 달리 랜덤한 파트가 없으나 비슷하게 모든 데이터를 매번 사용하지 않음
# 대신 매 스텝에서 미니 배치라는 임의의 작은 샘플 세트에 대해 그래디언트를 계산하여 감소하는 방향으로 파라미터 업데이트

# 장점 : 배치 경사 하강법에 비해 빠름, 랜덤한 파트 없어도 빠름, 미니 배치가 어느정도 크면 SGD에 비해 전역 최솟값에 가까이 도달
# 단점 : SGD에 비해 지역 최솟값에서 빠져나오기는 더 힘들 수 있음, 마찬가지로 하이퍼 파라미터 증가 

n_epoch = 50
minibatch_size = 20

np.random.seed(42)
B_hat_MGD = np.random.randn(2, 1) # 무작위 초기화

# learning schedule을 추가
t0, t1 = 200, 1000
def learning_schedule(t):
    return t0 / (t + t1)

# random.permutation : 배열이 저장된 변수자체를 변경하는 것이 아니라, 랜덤으로 섞은 배열만 반환
t = 0
for epoch in range(n_epoch):
    shuffled_indices = np.random.permutation(m) 
    X_b_shuffled = X_b[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    
    for i in range(0, m, minibatch_size):
        t += 1
        xi = X_b_shuffled[i : i + minibatch_size]
        yi = y_shuffled[i : i + minibatch_size]
        gradients = 2/minibatch_size * xi.T.dot(xi.dot(B_hat_MGD) - yi)
        eta = learning_schedule(t)
        B_hat_MGD = B_hat_MGD - eta * gradients

B_hat_MGD