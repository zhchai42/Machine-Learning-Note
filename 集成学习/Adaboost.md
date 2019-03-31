## AdaBoost

### 思想

提高那些前一轮弱分类器错误分类的样本的权值，降低正确分类的样本的权值。

使用加权表决结合弱分类器，加大分类误差率小的弱分类器的权值，减小误差率大的权值。

### 算法

对训练集
$$
T=\left\{\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right), \cdots,\left(x_{N}, y_{N}\right)\right\}
$$
其中$y_{i} \in \mathcal{Y}=\{-1,+1\}​$  使用弱分类器$G_i(x)​$分类，每次根据分类结果更新样本权值，重新训练分类器，训练$M​$次后，得到$M​$个弱分类器 $G_1(x),G_2(x),...,G_M(x)​$，将这些分类器加权加成。



#### 1. 初始化权值

对每个样本给定一个权值，权值分布$D_m​$可以看做样本出现的概率分布，初始分布$D_1​$假设为每个样本出现概率相同。
$$
D_{1}=\left(w_{11}, \cdots, w_{1 i}, \cdots, w_{1 N}\right), \quad w_{1 i}=\frac{1}{N}, \quad i=1,2, \cdots, N
$$

使用权值$w​$ 对样本集进行更新，它的影响体现在对分类错误率的计算上，而不对样本进行改变。

#### 2. 弱分类器训练 第$m$轮

使用分类器对概率分布为$D_m$样本训练，得到使得分类误差最低的模型$G_m(x)$

使用权值分布$D_m​$ 计算分类误差的期望：
$$
e_{m}=P\left(G_{m}\left(x_{i}\right) \neq y_{t}\right)=\sum_{i=1}^{N} w_{m i} I\left(G_{m}\left(x_{i}\right) \neq y_{i}\right)=\sum_{G_{m}\left(x_{i}\right) \neq y_{i}} w_{m i}
$$

#### 3. 计算分类器的系数 第$m$轮

$$
\alpha_{m}=\frac{1}{2} \ln \frac{1-e_{m}}{e_{m}}
$$

分类器系数$\alpha$的影响体现在

- 对权值的更新上
- 分类器集成时的加权

当错误率$e_m \leq \dfrac{1}{2}$时，$a_m \geq 0$ ，

当错误率大于$\dfrac{1}{2}​$，**<u>结束训练</u>**。

- 错误率强于随机猜测时，在更新权值时，分类正确，权值降低，分类错误，权值升高。

$a_m$随着$e_m​$减小而增大

- 错误率越小，样本权值更新时，对分类错误的样本越重视。
- 错误率越小，在最终分类器中所占权重越大。

#### 4. 更新样本的权值 第$m$轮

形成$D_{m+1}=\left(w_{m+1,1}, \cdots, w_{m+1, i}, \cdots, w_{m+1, N}\right)$
$$
w_{m+1, i}=\dfrac{w_{m i}}{Z_{m}} e^ \left(-\alpha_{m} y_{i} G_{m}\left(x_{i}\right)\right)
=\left\{\begin{array}{ll}{\dfrac{w_{m i}}{Z_{m}} \mathrm{e}^{-\alpha_{m}},} & {G_{m}\left(x_{i}\right)=y_{i}} \\ {\dfrac{w_{m i}}{Z_{m}} \mathrm{e}^{\alpha_{m}},} & {G_{m}\left(x_{i}\right) \neq y_{i}}\end{array}\right.
$$
其中$Z_m$为归一化因子，使得$D_{m+1}$作为概率分布，即$\sum_{i=1}^{N} w_{m+1, i}=1$。
$$
Z_{m}=\sum_{i=1}^{N} w_{m i} \exp \left(-\alpha_{m} y_{i} G_{m}\left(x_{i}\right)\right)
$$
可以看到当错误率$e_m \leq \dfrac{1}{2}$时，$a_m \geq 0​$ ，此时分类正确时，权值降低，分类错误时，权值升高。

权值升高和降低的倍率在特定的一轮更新中是相同的，仅和$\alpha​$有关，和样本无关。

#### 4.5 计算目前为止的总分类器

$m=1$时 $f_1(x) = \alpha_1 G_1(x)$

$m > 1$时 $f_m(x)=f_{m+1}(x)+\alpha_m G_m(x)$

#### 5.完成$M$轮训练后，加权集成

对于分类器$G_{m}(x) : \mathcal{X} \rightarrow\{-1,+1\}$，根据$\alpha $计算其加权线性组合。

$$
f(x)=\sum_{m=1}^{M} \alpha_{m} G_{m}(x)
$$
所有$\alpha​$的和并不为$1​$。

$f(x)$的绝对值体现了分类的置信度。

最终分类器为加权投票的结果：
$$
G(x)=\operatorname{sign}(f(x))=\operatorname{sign}\left(\sum_{m=1}^{M} \alpha_{m} G_{m}(x)\right)
$$














