### 高斯朴素贝叶斯 Gaussian Naive Bayes (GNB)

#### 定义

连续变量朴素贝叶斯的特殊情况

对于出如下情况：

- $Y$ 服从参数为$\pi$的伯努利分布（$Y$为二值）
- $X=<X_1, ... ,X_n>$ 属性均为连续变量

做出如下假设：

- 条件独立假设，每个属性给定对$Y​$独立。
- 每个属性的似然服从高斯分布，且类间方差相同（独立于$Y$），属性之间方差不同。而类间和属性之间的均值都不同。据此得到方差$\sigma_1, ... \sigma_n$共$n$个，均值$\mu_{ik}$ 一共$2n$个

#### 求后验概率

> PPT2 53

$$
\begin{aligned}
P(Y=1 | X)&=\frac{P(Y=1) P(X | Y=1)}{P(Y=1) P(X | Y=1)+P(Y=0) P(X | Y=0)} \\
&=\dfrac{1}{1+\dfrac{P(Y=0) P(X | Y=0)}{P(Y=1) P(X | Y=1)}} \\
&=\dfrac{1}{1+\exp \left(\ln \dfrac{P(Y=0) P(X | Y=0)}{P(Y=1) P(X | Y=1)}\right)} \\
&=\frac{1}{1+\exp \left(\ln \frac{P(Y=0)}{P(Y=1)}+\sum_{i} \ln \frac{P\left(X_{i} | Y=0\right)}{P\left(X_{i} | Y=1\right)}\right)} \\
&=\frac{1}{1+\exp \left(\ln \frac{1-\pi}{\pi}+\sum_{i} \ln \frac{P\left(X_{i} | Y=0\right)}{P\left(X_{i} | Y=1\right)}\right)}
\end{aligned}
$$

$$
\begin{aligned}
\ln \frac{P\left(X_{i} | Y=0\right)}{P\left(X_{i} | Y=1\right)} &= \ln\dfrac{\frac{1}{\sqrt{2\pi}\sigma_i}exp(-\frac{(x_i-\mu_{i0})^2}{2\sigma_i^2})}
{\frac{1}{\sqrt{2\pi}\sigma_i}exp(-\frac{(x_i-\mu_{i1})^2}{2\sigma_i^2})} \\
&=\ln exp(\dfrac{(x_i-\mu_{i1})^2-(x_i-\mu_{i0})^2}{2\sigma^2_i}) \\
&=\dfrac{2x_i(\mu_{i1}-\mu_{i0})+\mu_{i1}^2-\mu_{i0}^2}{2\sigma^2_i}
\end{aligned}
$$

$$
\begin{aligned}
P(Y=1 | X)
&=\frac{1}{1+\exp \left(\ln \frac{1-\pi}{\pi}+\sum_{i} \ln \frac{P\left(X_{i} | Y=0\right)}{P\left(X_{i} | Y=1\right)}\right)} \\
&=\frac{1}{1+\exp \left(\ln \frac{1-\pi}{\pi}+\sum_{i}\left(\frac{\mu_{i 0}-\mu_{i 1}}{\sigma_{i}^{2}} X_{i}+\frac{\mu_{i 1}^{2}-\mu_{i 0}^{2}}{2 \sigma_{i}^{2}}\right)\right)} \\
&=\frac{1}{1+\exp \left(w_{0}+\sum_{i=1}^{n} w_{i} X_{i}\right)} \\
&= \frac{1}{1+e^{-\theta^{T} x_{n}}}
\end{aligned}
$$

其中 $w_{i}=\dfrac{\mu_{i 0}-\mu_{i 1}}{\sigma_{i}^{2}}$ 每个属性不同， $w_{0}=\ln \dfrac{1-\pi}{\pi}+\sum_{i} \dfrac{\mu_{i 1}^{2}-\mu_{i 0}^{2}}{2 \sigma_{i}^{2}}​$ 对每个属性相同。

同时：
$$
P(Y=0 | X)=1-P(Y=1 | X)=\frac{\exp \left(w_{0}+\sum_{i=1}^{n} w_{i} X_{i}\right)}{1+\exp \left(w_{0}+\sum_{i=1}^{n} w_{i} X_{i}\right) = }\frac{e^{-\theta^{T} x_{n}}}{1+e^{-\theta^{T} x_{n}}}
$$

#### 决策

比较两个后验概率，可以将它们相比，取对数，即求对数几率：
$$
\ln \dfrac{P(Y=1 | X)}{P(Y=0 | X)}=\ln \dfrac{ \frac{1}{1+e^{-\theta^{T} x_{n}}}}{\frac{e^{-\theta^{T} x_{n}}}{1+e^{-\theta^{T} x_{n}}}} =\theta^{T} x_{n}
$$

#### 多分类GNB

$$
p\left(y_{n}^{k}=1 | x_{n}\right)=\frac{e^{-\theta_{k}^{T} x_{n}}}{\sum_{j} e^{-\theta_{j}^{T} x_{n}}}
$$

等价于softmax 函数



#### 与LR的对比

|        |                             GNB                              | LR                                                 |
| ------ | :----------------------------------------------------------: | -------------------------------------------------- |
| 判别式 | 根据$\dfrac{P(Y=1|X)}{P(Y=0|X)}=\theta^{T} x_{n}$ 判断，$P(Y=1|X)=\dfrac{1}{1+e^{-\theta^{T} x}}$ | $y=\dfrac{1}{1+e^{-\theta^{T} x}}$ 直接得到y的分类 |
| 模型   |                          生成式模型                          | 判别式模型                                         |
| 原理   |       估计$P(X|Y)$  $P(Y)$  使用贝叶斯公式得到$P(Y|X)$       | 设定一个假设函数直接得出$P(Y|X)$ 直接估计$P(Y|X)$  |



























