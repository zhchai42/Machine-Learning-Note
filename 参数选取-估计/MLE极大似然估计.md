## 极大似然估计 Maximum Likelihood Estimation

##### #估计方法   #频率派方法

### 原理

- 假设模型确定，选择模型参数$\theta$使得当前数据$D$出现的概率$P(D|\theta)$最大。

$$
\hat{\theta}_{M L E}=\arg \max _{\theta} P(D | \theta)
$$

- 不预设模型的参数的先验分布，不具备任何有关先验的知识，即先验分布为均匀分布的MAP。

### 缺点

- 在数据缺乏的情况下表现不佳。

### 抛硬币

> 抛$\alpha_T+\alpha_H$次硬币，$\alpha_H$次为头H，$\alpha_T$次为Tail 估计硬币头朝上的概率

#### 模型

满足参数为$\theta$ 的伯努利分布 $P(head)=\theta$   $P(tail)=1-\theta$

每次抛硬币相互独立

#### 似然函数

$$
\begin{aligned} 
 P(D | \theta) &= \prod_{i=1}^{n} P\left(X_{i} | \theta\right) 每次抛硬币独立\\  
&= \prod_{i : X_{i}=H} \theta \prod_{i : X_{i}=T}(1-\theta) \\ 
&=\theta^{\alpha_{H}}(1-\theta)^{\alpha_{T}}\\ 

\end{aligned}
$$

#### 模型参数优化

$$
\begin{aligned} 
\hat{\theta}_{M L E} &=\arg \max _{\theta} P(D | \theta) \\ 
&=\arg \max _{\theta} \theta^{\alpha_{H}}(1-\theta)^{\alpha_{T}}\\ 
&=\arg \max _{\theta} J(\theta)
\end{aligned}
$$

#### 导数法优化

$$
\begin{aligned} 
\frac{\partial J(\theta)}{\partial \theta} 
&=\alpha_{H} \theta^{\alpha_{H}-1}(1-\theta)^{\alpha_{T}}-\alpha_{T} \theta^{\alpha_{H}}(1-\theta)^{\alpha_{T}-1} \\ &=\theta^{\alpha_{H}-1}(1-\theta)^{\alpha_{T}-1}\left(\alpha_{H}(1-\theta)-\alpha_{T} \theta\right) \\
&=0
\end{aligned}
$$

$$
\begin{aligned}
\theta^{\alpha_{H}-1}(1-\theta)^{\alpha_{T}-1}\left(\alpha_{H}(1-\theta)-\alpha_{T} \theta\right)  &= 0\\
\alpha_{H}(1-\theta)-\alpha_{T} \theta &= 0\\
\hat{\theta}_{M L E}&=\frac{\alpha_{H}}{\alpha_{H}+\alpha_{T}}
\end{aligned}
$$

得到的$\theta$值等于抛出head的频率。









































