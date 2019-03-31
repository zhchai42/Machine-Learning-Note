## 对数几率回归 Logistic Regression

为什么LR的决策面为直线。

### 定义

在线性回归上套一个sigmoid函数，进行二分类。
$$
h_{\theta}(x)=g\left(\theta^{T} x\right)=\frac{1}{1+e^{-\theta^{T} x}}
$$
其中$g(z)=\dfrac{1}{1+e^{-z}}$被称为logistic函数。

### Logistic函数的性质

#### 导数

$$
\begin{aligned} g^{\prime}(z) &=\frac{d}{d z} \frac{1}{1+e^{-z}} \\ &=\frac{1}{\left(1+e^{-z}\right)^{2}}\left(e^{-z}\right) \\ &=\frac{1}{\left(1+e^{-z}\right)} \cdot\left(1-\frac{1}{\left(1+e^{-z}\right)}\right) \\ &=g(z)(1-g(z)) \end{aligned}
$$

### 概率假设下的模型

假设$y​$服从伯努利分布$y|x ; \theta \sim​$Bernoulli$(\phi)​$， $h(x)​$得出的是其为正例的概率，即：
$$
\begin{array}{l}{P(y=1 | x ; \theta)=h_{\theta}(x)} \\ {P(y=0 | x ; \theta)=1-h_{\theta}(x)}\end{array}
$$
或
$$
p(y | x ; \theta)=\left(h_{\theta}(x)\right)^{y}\left(1-h_{\theta}(x)\right)^{1-y}
$$

### 学习参数 极大似然估计

概率假设下，得到似然函数：
$$
\begin{aligned} L(\theta) &=p(\vec{y} | X ; \theta) \\ &=\prod_{i=1}^{m} p\left(y^{(i)} | x^{(i)} ; \theta\right) \\ &=\prod_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)\right)^{y^{(i)}}\left(1-h_{\theta}\left(x^{(i)}\right)\right)^{1-y^{(i)}} \end{aligned}
$$
对数似然为
$$
\begin{aligned} l(\theta) &=\log L(\theta) \\ &=\sum_{i=1}^{m} (y^{(i)} \log h\left(x^{(i)}\right)+\left(1-y^{(i)}\right) \log \left(1-h\left(x^{(i)}\right)\right)) \end{aligned}
$$
称为交叉熵 cross entropy

#### 优化  损失函数

$$
\operatorname{argmax}_{\theta} \sum_{i=1}^{m} (y^{(i)} \log h_{\theta}\left(x^{(i)}\right)+\left(1-y^{(i)}\right) \log \left(1-h_{\theta}\left(x^{(i)}\right)\right))
$$

#### 梯度上升 gradient ascent

$$
\theta :=\theta+\alpha \nabla_{\theta} l(\theta)
$$

求梯度：
$$
\begin{aligned} \frac{\partial}{\partial \theta_{j}} l(\theta) &=\left(y \frac{1}{g\left(\theta^{T} x\right)}-(1-y) \frac{1}{1-g\left(\theta^{T} x\right)}\right) \frac{\partial}{\partial \theta_{j}} g\left(\theta^{T} x\right) \\ &=\left(y \frac{1}{g\left(\theta^{T} x\right)}-(1-y) \frac{1}{1-g\left(\theta^{T} x\right)}\right) g\left(\theta^{T} x\right)\left(1-g\left(\theta^{T} x\right)\right) \frac{\partial}{\partial \theta_{j}} \theta^{T} x \\ 
&=\left(y\left(1-g\left(\theta^{T} x\right)\right)-(1-y) g\left(\theta^{T} x\right)\right) x_{j} \\ 
&=\left(y(1-h(x))-(1-y) h(x)\right) x_{j} \\
&=\left(y-h_{\theta}(x)\right) x_{j} 
\end{aligned}
$$
则梯度上升更新为：
$$
\theta_{j} :=\theta_{j}+\alpha\left(y^{(i)}-h_{\theta}\left(x^{(i)}\right)\right) x_{j}^{(i)}
$$
可以发现类似为LMS，只不过$h(x)$不同

#### 牛顿法 

具体见 牛顿法.md

对对数似然的相反数求极小值：
$$
\arg \min _{\theta} J(\theta)=\frac{1}{m} \sum_{i=1}^{m}(-y^{(i)} \log h_{\theta}\left(x^{(i)}\right)-\left(1-y^{(i)}\right) \log \left(1-h_{\theta}\left(x^{(i)}\right)\right))
$$
梯度向量（推导见上）：
$$
\nabla_{\theta} J(\theta)=\frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) x_{j}
$$
Hessian 矩阵
$$
H=\frac{1}{m} \sum_{i=1}^{m} h_{\theta}\left(x^{(i)}\right)\left(1-h_{\theta}\left(x^{(i)}\right)\right) x^{(i)}\left(x^{(i)}\right)^{T}
$$
更新：
$$
\theta^{(t+1)}=\theta^{(t)}-H^{-1} \nabla J\left(\theta^{(t)}\right)
$$
