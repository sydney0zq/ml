##Neural Networks: Representation

###Non-linear Hypotheses

Performing linear regression with a complex set of data with many features is very unwieldy. Say you wanted to create a hypothesis from three (3) features that included all the quadratic terms:

$$
\begin{align*}
& g(\theta_0 + \theta_1x_1^2 + \theta_2x_1x_2 + \theta_3x_1x_3 \newline
& + \theta_4x_2^2 + \theta_5x_2x_3 \newline
& + \theta_6x_3^2 )
\end{align*}
$$

That gives us $6$ features. The exact way to calculate how many features for all polynomial terms is [http://www.mathsisfun.com/combinatorics/combinations-permutations.html the combination function with repetition]: $\frac{(n+r-1)!}{r!(n-1)!}$. In this case we are taking all two-element combinations of three features: $\frac{(3 + 2 - 1)!}{(2!\cdot (3-1)!)} $=$\frac{4!}{4} = 6$. 

For 100 features, if we wanted to make them quadratic we would get $\frac{(100 + 2 - 1)!}{(2\cdot (100-1)!)} = 5050$ resulting new features.

We can approximate the growth of the number of new features we get with all quadratic terms with $\mathcal{O}(n^2/2)$. And if you wanted to include all cubic terms in your hypothesis, the features would grow asymptotically at $\mathcal{O}(n^3)$. These are very steep growths, so as the number of our features increase, the number of quadratic or cubic features increase very rapidly and becomes quickly impractical.

Example: let our training set be a collection of 50x50 pixel black-and-white photographs, and our goal will be to classify which ones are photos of cars. Our feature set size is then $n=2500$ if we compare every pair of pixels.

Now let's say we need to make a quadratic hypothesis function. With quadratic features, our growth is $\mathcal{O}(n^2 / 2)$. So our total features will be about $2500^2 / 2 = 3125000$, which is very impractical.

Neural networks offers an alternate way to perform machine learning when we have complex hypotheses with many features.

####Neurons and the Brain

Neural networks are limited imitations of how our own brains work. They've had a big recent resurgence because of advances in computer hardware.

There is evidence that the brain uses only one "learning algorithm" for all its different functions. Scientists have tried cutting (in an animal brain) the connection between the ears and the auditory cortex and rewiring the optical nerve with the auditory cortex to find that the auditory cortex literally learns to see.

####Model Representation I



























































































