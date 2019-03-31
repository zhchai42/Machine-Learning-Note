## 广义线性模型 Generalized Linear Model (GLM)

### 背景

对于线性回归，可以看做 $y|x ; \theta \sim N\left(\mu, \sigma^{2}\right)$ 

对于Logistic回归，可以看做$y|x ; \theta \sim$$Bernoulli(\phi)$

其中$\mu$ $\phi$ 为 $x$ 和 $\theta$的某种函数

#### 指数族  The exponential family

##### 指数族分布  exponential family distributions

如果一个分布能用下面的方式来写出来，我们就说这类分布属于指数族：
$$
p(y ; \eta)=b(y) e^{\eta^{T} T(y)-a(\eta)}
$$

其中$\eta$ 叫做此分布的**自然参数** （natural parameter，也叫**典范参数 canonical parameter**） 

$T(y)$叫做**充分统计量（sufficient statistic）**

$ a(\eta)$ 是一个**对数分割函数（log partition function）**

 $e^{−a(\eta)}$ 这个量本质上扮演了归一化常数（normalization constant）的角色，也就是确保 $p(y; \eta)$的总和或者积分等于1

$y$又称响应变量 response variable

当给定 $T$, $a$和 $b$ 时，就定义了一个用 $\eta$ 进行参数化的分布族（family，或者叫集 set）；通过改变 $\eta$，我们就能得到这个分布族中的不同分布

##### 伯努利分布$Bernoulli(\phi)$是以$\phi$的函数为自然参数的指数族分布

$$
\begin{aligned} p(y ; \phi) &=\phi^{y}(1-\phi)^{1-y} \\ &=\exp (y \log \phi+(1-y) \log (1-\phi)) \\ &=\exp \left(\left(\log \left(\frac{\phi}{1-\phi}\right)\right) y+\log (1-\phi)\right) \end{aligned}
$$

其中
$$
\begin{aligned} 
\eta&=\log \left(\frac{\phi}{1-\phi}\right)\\
\phi&=1 /\left(1+e^{-\eta}\right) \\
T(y) &=y \\ a(\eta) &=-\log (1-\phi) \\ &=\log \left(1+e^{\eta}\right) \\ b(y) &=1 \end{aligned}
$$

##### 高斯分布是指数族分布

假设方差为1，有：
$$
\begin{aligned} p(y ; \mu) &=\frac{1}{\sqrt{2 \pi}} \exp \left(-\frac{1}{2}(y-\mu)^{2}\right) \\ &=\frac{1}{\sqrt{2 \pi}} \exp \left(-\frac{1}{2} y^{2}\right) \cdot \exp \left(\mu y-\frac{1}{2} \mu^{2}\right) \end{aligned}
$$
其中：
$$
\begin{aligned} \eta &=\mu \\ T(y) &=y \\ a(\eta) &=\mu^{2} / 2 \\ &=\eta^{2} / 2 \\ b(y) &=(1 / \sqrt{2 \pi}) \exp \left(-y^{2} / 2\right) \end{aligned}
$$

##### 其他指数族分布

- 多项式分布
- 泊松分布
- 伽马分布 指数分布
- 贝塔分布 狄利克雷分布 等

### 定义

#### 构建广义线性模型

构建广义线性模型的目的是已知或者假设 $y$服从某种分布  $P(y|x;\theta)$ ，我们需要知道如何设计它的假设函数$y=h(x)$ 用来拟合$y$.

- 给定 $x$ 和$ \theta$ ,$y$ 的分布$P(y|x;\theta )$属于指数分布族，即将$P(y|x;\theta)$转变为指数分布族的形式

$y|x ; \theta \sim \text {ExponentialFamily}(\eta)$


- 假设函数$h(x)$为$T(y)$的期望 ，即在该形式中对$T(y)$求期望，就能得到假设函数。

$$
h_\theta(x)=E[T(y) | x]
$$

有$g(\eta)=E[T(y) ; \eta]$ 被称为规范相应函数 **canonical response function** 

对于大多数情况，$T(y)=y$
$$
h(x)=E[y | x]
$$
以逻辑回归为例：
$$
h_{\theta}(x)=[p(y=1 | x ; \theta ) ]=[0 \cdot p(y=0 | x ; \theta)+1 \cdot p(y=1 | x ; \theta)]=E[y | x ; \theta]
$$

- 自然参数和$x$线性相关，$\eta=\theta^{T} x$，或者如果 $\eta$ 是有值的向量，则有$\eta_i = \theta_i^T x$ 

把$g(\eta)=E[T(y) ; \eta]$ 中的$\eta$替换成$\theta^Tx$ ，就得到了假设函数 $h_\theta(x)$ 

#### 实例

已知$y$服从伯努利分布 $y \sim Bernoulli(\phi)$求$y$的假设函数：

- 把伯努利分布转化为指数分布族 

$$
\begin{aligned} p(y ; \phi) &=\phi^{y}(1-\phi)^{1-y} \\ &=\exp (y \log \phi+(1-y) \log (1-\phi)) \\ &=\exp \left(\left(\log \left(\frac{\phi}{1-\phi}\right)\right) y+\log (1-\phi)\right) \end{aligned}
$$

- 得到$\eta$ 和 $g(\eta)=E[T(y) ; \eta]$

$$
\begin{aligned}
\eta&=\log \left(\frac{\phi}{1-\phi}\right)\\
T(y)&=y\\
g(\eta)&=E[T(y) ; \eta]=\phi \times 1 + (1-\phi) \times 0 = \phi=\dfrac{1}{\left(1+e^{-\eta}\right)}
\end{aligned}
$$

- 代入假设 $\eta =\theta^Tx$

$$
h_\theta(x)=g(\theta^Tx)=\dfrac{1}{\left(1+e^{-\theta^Tx}\right)}
$$

由此我们得出，如果$y$服从伯努利分布，基于广义线性模型进行拟合，可以得到logistic回归。

同理可以得到， 如果$y$服从高斯分布，基于GLM拟合，可以得到线性回归。

#### 性质

$$
E(y | \eta)=\frac{d}{d \eta} a(\eta) \\
\operatorname{Var}(y | \eta)=\frac{d^{2}}{d \eta^{2}} a(\eta)
$$

可得假设函数为：
$$
h(x, \theta)=E(y | x, \theta)=\frac{d}{d \eta} a(\eta)
$$




























