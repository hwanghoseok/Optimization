
# 최소제곱법(ordinary least squares)  
  


#### 해방적식을 근사적으로 구하는 방법, 근사적으로 구하려는 해와 실제 해의 오차의 제곱의 합이 최소가 되는 해를 구하는 방법 
<br>




> 한계

<br>

- 노이즈(outlier)에 취약
- 특정한 변수와 샘플 크기에 비례해 계산 비용이 높음

<br>

> RSS(Resudual Sum of Square) 공식 

<br>

- 실제 값(y)과 가설(y_hat) 예측 값의 차이가 가장 작은 계수 계산 
$$ 
y = wx + b  
$$
$$ 
\hat{w} = \frac{\sum_{}(x-\overline{x})(y-\overline{y})}{\sum_{}(x-\overline{x})^2}
$$
$$
b = \overline{y} - a\overline{x}
$$

```python 
def make_linear(w = 0.3, b = 0.8, size = 50, noise = 1):
    x = np.arange(size) # 샘플 50개 생성
    y = w*x + b
    noise = np.random.uniform(-abs(noise),abs(noise),size = y.shape) # noise 생성
    yy = y + noise # 추가
    plt.figure(figsize = (10,7))
    plt.plot(x, y colore = 'r', label = f'y = {w}*x + {b}')
    plt.scatter(x,yy, label = 'data')
    plt.legend(fontsize = 20)
    plt.show()
    print(f'w: {w}, b: {b}')
    return x, yy
```
