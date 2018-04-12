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

