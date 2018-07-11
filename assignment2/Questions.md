Need to use tensor board to see

2-a

| stack        		| buffer                                 | new dependency           | transition        |
|-------------------|:---------------------------------------|:--------------------------------------:|----------------:|
| [ROOT]        		| [I, parsed, this, sentence, correctly] |       	| Initial Configuration |
| [ROOT, I]      		| [parsed, this, sentence, correctly]                                |       | SHIFT   |
| [ROOT, I, parsed] | [this, sentence, correctly]                                |       | SHIFT   |
| [ROOT, parsed]    | [this, sentence, correctly]                               | parsed $\rightarrow$ I      | LEFT-ARC   |

2-f:

$E_{p_{drop}}[h_{drop}]_i = r(1-p_{drop})h_i$
So $r=\frac{1}{1-p_{drop}}$

2-g-i:
The momentum $m$ makes the updates from varying as much, because it is combined with the original momentum and current gradient. So it could regulate the update extent by modifying $\beta_{1}$. 
It might help with learning when the algorithm encounters a suddenly changed gradient. In this case, with original SGD, the algorithm will jump to a faraway point, but with Adam it can go to a nearer place, and prevent changing so much.

Right answer:
Each update will be mostly the same as the previous one (only $1-\beta_1$ of m changes
each step), so the updates won't vary as much. One way of thinking about this is that it will stop
the model parameters from "bouncing around as much" when moving towards a local optimum.
Another way is that doing the rolling average is a bit like computing the gradient over a larger
minibatch, so each update will be closer to the true gradient over the whole dataset (i.e., lower
variance means each gradient estimate is closer to the mean).

2-g-ii:
Solution: 
The parameters with the smallest gradients (on average) will get the larger updates.
This means parameters that are at a place where the loss with respect to them is pretty at will get larger updates, helping them move off plateaus.

3.Recurrent Neural Networks: Language Modeling
a. as $y^{(t)}$ is one-hot, we have $J^{(t)}(\theta) = log PP^{(t)}$

for $|V|=10000$, we have 
$$
PP = 10000 \\
CE = log10000 = 9.21 
$$

b. For cross entropy loss, according to the solution of assignment 1, we have
$$\frac{\partial J^{(t)}}{\partial \theta^{(t)}} = {\hat{y}}^{(t)} - y^{(t)}$$ 
where $\theta^{(t)}=h_{(t)}U+b_2$
So other derivatives are easy to get as below:
$$
\frac{\partial J^{(t)}}{\partial b_2} = {\hat{y}}^{(t)} - y^{(t)}  \\
\frac{\partial J^{(t)}}{\partial h_{(t)}} = \frac{\partial J^{(t)}}{\partial \theta^{(t)}}U^{T}  
$$
as $x^{(t)}$ is one-hot vector,
$$
\frac{\partial J^{(t)}}{\partial L_{x^{(t)}}} = (\frac{\partial J^{(t)}}{\partial h_{(t)}} \circ (1 - h_{(t)}) \circ h_{(t)})I^{T}\\
$$
We also have
$$
\frac{\partial J^{(t)}}{\partial I}\Bigr|
_{(t)} = {e^{(t)}}^T (\frac{\partial J^{(t)}}{\partial h_{(t)}} \circ (1 - h_{(t)}) \circ h_{(t)}) \\
\frac{\partial J^{(t)}}{\partial H}\Bigr|
_{(t)} = {h^{(t-1)}}^T (\frac{\partial J^{(t)}}{\partial h_{(t)}} \circ (1 - h_{(t)}) \circ h_{(t)})\\
\frac{\partial J^{(t)}}{\partial h^{(t-1)}} =(\frac{\partial J^{(t)}}{\partial h_{(t)}} \circ (1 - h_{(t)}) \circ h_{(t)}) H^T\\
$$

c. As we already have $\frac{\partial J^{(t)}}{\partial h^{(t-1)}}$, we can write the similar derivatives:
$$
\frac{\partial J^{(t)}}{\partial L_{x^{(t-1)}}} = (\frac{\partial J^{(t)}}{\partial h_{(t-1)}} \circ (1 - h_{(t-1)}) \circ h_{(t-1)})I^{T}\\
\frac{\partial J^{(t)}}{\partial I}\Bigr|
_{(t-1)} = {e^{(t-1)}}^T (\frac{\partial J^{(t)}}{\partial h_{(t-1)}} \circ (1 - h_{(t-1)}) \circ h_{(t-1)}) \\
\frac{\partial J^{(t)}}{\partial H}\Bigr|
_{(t-1)} = {h^{(t-2)}}^T (\frac{\partial J^{(t)}}{\partial h_{(t-1)}} \circ (1 - h_{(t-1)}) \circ h_{(t-1)})
$$

d.
Forward propagation:
$O(D_h*D_h + d*D_h + |V|*D_h + a*D_h + b*|V|) = O((D_h + d + |V|) * D_h) \approx O(|V| * D_h)$ 
Back propagation:
For all derivatives in b:
$(O(|V| + D_h*|V| + 2*D_h + d*D_h + D_h * D_h + D_h * D_h)) \approx O(|V| * D_h)$
For $\tau$ steps:
$(O(|V| + D_h*|V| + 2*D_h + \tau * (d*D_h  + D_h * D_h + D_h * D_h)))$
We can see that the most time-consuming step is decoding and its back propagation. That is, translate the hidden vector into vocabulary dim vector step, as vocabulary size could be very large(assuming $|v| >> Dh$).


