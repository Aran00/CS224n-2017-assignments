1. Softmax: Omitted
2. Neural Network Basics
 a. $\frac{\partial \sigma(x)}{\partial x} = -\frac{-e^{-x}}{(1+e^{-x})^2} = \sigma(x)(1-\sigma(x))$
 
 b. And only k-th dimension of $\boldsymbol{y}$ is 1, so
	$$s = CE(\boldsymbol{y}, \boldsymbol{\hat{y}}) = - \sum_{i} y_i log(\hat{y_i}) = -log(\hat{y_k})$$
	As $\boldsymbol{\hat{y}} = softmax(\boldsymbol{\theta})$, we have
	$$\hat{y_k}=softmax(\boldsymbol{\theta})_k = \frac{e^{\boldsymbol{\theta}_{k}}}{\sum_{j}{e^{\boldsymbol{\theta}_j}}}$$
Then 
	$$\frac{\partial s}{\partial \theta_i} = \frac{\partial{s}}{\partial \hat{y_k}}\frac{\partial \hat{y_k}}{\partial \theta_i}=-\frac{1}{\hat{y_k}} \frac{\partial \hat{y_k}}{\partial \theta_i} $$
In which 
$$ \frac{\partial \hat{y_k}}{\partial \theta_i} = 
\begin{cases} 
	\hat{y_i} (1-\hat{y_i} ) & i = k \\
	-\hat{y_k}\hat{y_i} & i \neq k
\end{cases}
$$
So the last result is
$$\frac{\partial s}{\partial \boldsymbol{\theta}}=\boldsymbol{\hat{y}} - \boldsymbol{y}$$ 
Surprisingly simple...

 c. I think it's good to regard $\boldsymbol{x}$ as a row vector...
 As 
 $$
 J = CE(\boldsymbol{y}, \boldsymbol{\hat{y}}) \\ 
 \boldsymbol{\hat{y}} = softmax(\boldsymbol{\theta}) \\
 \boldsymbol{\theta} = \boldsymbol{hW_2 + b_2} \\
 \boldsymbol{h} = sigmoid(\boldsymbol{a}) \\
 \boldsymbol{a} = \boldsymbol{xW_1+b_1}
 $$
 where there dimensions are:
 $$
 \boldsymbol{x} : 1 * D_x \\
 W_1: D_x * H \\
 \boldsymbol{a, b_1, h}: 1 * H \\
 W_2: H * D_y \\
 \boldsymbol{\theta, b_2, \hat{y}}: 1 * D_y
 $$
 Let's play a dangerous game, only care about dimension:
 $$
 \frac{\partial J}{\partial \boldsymbol{x}} = \frac{\partial J}{\partial \boldsymbol{a}} {W_1^T} = \sigma^{\prime}(\boldsymbol{a}) \circ \frac{\partial J}{\partial \boldsymbol{h}} {W_1^T} \\
 = (\sigma^{\prime}(\boldsymbol{a}) \circ \frac{\partial J}{\partial \boldsymbol{\theta}}{W_2^T}) {W_1^T} \\
 = (\sigma(\boldsymbol{a})(1-\sigma(\boldsymbol{a})) \circ (\boldsymbol{\hat{y}} - \boldsymbol{y}){W_2^T}) {W_1^T}
 $$
 Where $\circ$ means element wise multiply.
 
 d. Considering $W$ and $\boldsymbol{b}$, we have 
 $$
 (D_x+1)*H + D_y(1+H)
 $$
 parameters in this NN.
 
 e-g: See in code.
 
3.  

 



