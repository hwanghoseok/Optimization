
###### 최소제곱법(ordinary least squares)  
# 해방적식을 근사적으로 구하는 방법, 근사적으로 구하려는 해와 실제 해의 오차의 제곱의 합이 최소가 되는 해를 구하는 방법 

# 한계

# 1.노이즈(outlier)에 취약
# 2.특정한 변수와 샘플 크기에 비례해 계산 비용이 높음

import numpy as np
import matplotlib.pyplot as plt

def make_linear(w = 0.3, b = 0.8, size = 50, noise = 1):
    x = np.arange(size) # 샘플 50개 생성
    y = w*x + b
    noise = np.random.uniform(-abs(noise),abs(noise),size = y.shape) # noise 생성
    yy = y + noise # 추가
    plt.figure(figsize = (10,7))
    plt.plot(x, y, color = 'r', label = f'y = {w}*x + {b}')
    plt.scatter(x,yy, label = 'data')
    plt.legend(fontsize = 20)
    plt.show()
    print(f'w: {w}, b: {b}')
    return x, yy

x, y = make_linear(size = 50, w = 1.5, b = 0.8, noise = 5.5)


# noise 값을 증가 시켰을 때
x, y = make_linear(size = 100, w = 0.7, b = 0.2, noise = 5.5)

# 임의로 2개의 outlier 추가
y[5] = 60
x[10] = 60

plt.figure(figsize = (10, 7))
plt.scatter(x, y)
plt.show() 

# outlier에 취약