2-a

| stack        		| buffer                                 | new dependency           | transition        |
|-------------------|:---------------------------------------|:--------------------------------------:|----------------:|
| [ROOT]        		| [I, parsed, this, sentence, correctly] |       	| Initial Configuration |
| [ROOT, I]      		| [parsed, this, sentence, correctly]                                |       | SHIFT   |
| [ROOT, I, parsed] | [this, sentence, correctly]                                |       | SHIFT   |
| [ROOT, parsed]    | [this, sentence, correctly]                               | parsed $\rightarrow$ I      | LEFT-ARC   |


