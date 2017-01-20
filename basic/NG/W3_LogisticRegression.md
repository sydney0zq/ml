[TOC]

##Logistic Regression

Now we are switching from regression problems to **classification problems**. Don't be confused by the name "Logistic Regression"; it is named that way for historical reasons and is actually an approach to classification problems, not regression problems.

###Binary Classification

Instead of our output vector $y$ being a continuous range of values, it will only be 0 or 1.

$$
y \in \lbrace 0,1 \rbrace
$$

Where 0 is usually taken as the "negative class" and 1 as the "positive class", but you are free to assign any representation to it.

We're only doing two classes for now, called a "Binary Classification Problem."

One method is to use linear regression and map all predictions greater than 0.5 as a 1 and all less than 0.5 as a 0.  **This method doesn't work well because classification is not actually a linear function.**

###Hypothesis Representation

Our hypothesis should satisfy:

$$
0 \leq h_\theta (x) \leq 1
$$

Our new form uses the "Sigmoid Function," also called the "Logistic Function":

$$
\begin{align*}
& h_\theta (x) =  g ( \theta^T x ) \newline \newline
& z = \theta^T x \newline
& g(z) = \dfrac{1}{1 + e^{-z}}
\end{align*}
$$

![Logistic function](../SRC/ml_ch3_0.png)

The function $g(z)$, shown here, maps any real number to the (0, 1) interval, making it useful for transforming an arbitrary-valued function into a function better suited for classification. Try playing with interactive plot of sigmoid function <https://www.desmos.com/calculator/bgontvxotm>.

We start with our old hypothesis (linear regression), except that we want to restrict the range to 0 and 1. This is accomplished by plugging $\theta^T x $ into the Logistic Function.

$h_\theta$ will give us the **probability** that our output is 1. For example, $h_\theta(x)=0.7$ gives us the probability of 70% that our output is 1.

$$
\begin{align*}
& h_\theta(x) = P(y=1 | x ; \theta) = 1 - P(y=0 | x ; \theta) \newline
& P(y = 0 | x;\theta) + P(y = 1 | x ; \theta) = 1
\end{align*}
$$

Our probability that our prediction is 0 is just the complement of our probability that it is 1 (e.g. if probability that it is 1 is 70%, then the probability that it is 0 is 30%).

###Decision Boundary

In order to get our discrete 0 or 1 classification, we can translate the output of the hypothesis function as follows:

$$
\begin{align*}
& h_\theta(x) \geq 0.5 \rightarrow y = 1 \newline
& h_\theta(x) < 0.5 \rightarrow y = 0 \newline
\end{align*}
$$

The way our logistic function $g$ behaves is that when its input is greater than or equal to zero, its output is greater than or equal to 0.5:

$$
\begin{align*}
& g(z) \geq 0.5 \newline
& when \; z \geq 0
\end{align*}
$$

Remember.-

$$
\begin{align*}
z=0,  e^{0}=1 \Rightarrow  g(z)=1/2\newline 
z \to \infty, e^{-\infty} \to 0 \Rightarrow g(z)=1 \newline
 z \to -\infty, e^{\infty}\to \infty \Rightarrow g(z)=0 
\end{align*}
$$

So if our input to $g$ is $\theta^T X$, then that means:

$$
\begin{align*}
& h_\theta(x) = g(\theta^T x) \geq 0.5 \newline
& when \; \theta^T x \geq 0
\end{align*}
$$

From these statements we can now say:

$$
\begin{align*}
& \theta^T x \geq 0 \Rightarrow y = 1 \newline
& \theta^T x < 0 \Rightarrow y = 0 \newline
\end{align*}
$$

The **decision boundary** is the line that separates the area where y=0 and where y=1. **It is created by our hypothesis function. You should note that the decision boundary is a property, not of the trading set, but of the hypothesis under the parameters.**

Again, the input to the sigmoid function $g(z)$ (e.g. $\theta^T X $) doesn't need to be linear, and could be a function that describes a circle (e.g. $z = \theta_0 + \theta_1 x_1^2 +\theta_2 x_2^2 $) or any shape to fit our data.

###Cost Function

We cannot use the same cost function that we use for linear regression because the Logistic Function will cause the output to be wavy, causing many local optima. In other words, it will not be a convex function.

![Trend of cost function with linear regression](../SRC/ml_ch3_1.png)

Instead, our cost function for logistic regression looks like:

$$
\begin{align*}
& J(\theta) = \dfrac{1}{m} \sum_{i=1}^m \mathrm{Cost}(h_\theta(x^{(i)}),y^{(i)}) \newline
& \mathrm{Cost}(h_\theta(x),y) = -\log(h_\theta(x)) \; & \text{if y = 1} \newline
& \mathrm{Cost}(h_\theta(x),y) = -\log(1-h_\theta(x)) \; & \text{if y = 0}
\end{align*}
$$

![](../SRC/ml_ch3_2.png)

The more our hypothesis is off from y, the larger the cost function output. If our hypothesis is equal to y, then our cost is 0:

$$
\begin{align*}
& \mathrm{Cost}(h_\theta(x),y) = 0 \text{  if  } h_\theta(x) = y \newline
& \mathrm{Cost}(h_\theta(x),y) \rightarrow \infty \text{  if  } y = 0 \; \mathrm{and} \; h_\theta(x) \rightarrow 1 \newline
& \mathrm{Cost}(h_\theta(x),y) \rightarrow \infty \text{  if  } y = 1 \; \mathrm{and} \; h_\theta(x) \rightarrow 0 \newline
\end{align*}
$$

If our correct answer 'y' is 0, then the cost function will be 0 if our hypothesis function also outputs 0. If our hypothesis approaches 1, then the cost function will approach infinity.

If our correct answer 'y' is 1, then the cost function will be 0 if our hypothesis function outputs 1. If our hypothesis approaches 0, then the cost function will approach infinity.

Note that writing the cost function in this way guarantees that $J(\theta)$ is convex for logistic regression.

Take an example for better understanding: Suppose we have already a cost function which predicts a tumor is malignant or not. As long as we correctly predict the result ( $h_\theta(x)\rightarrow 1$ and $y\rightarrow1$), the cost is zero($Cost\rightarrow0$). But also note that if we mistakely predict the result $y$($h_\theta\rightarrow0$ and $y\rightarrow1$), then **we penalize the learning algorithm by a large cost.**

###Simplified Cost Function and Gradient Descent

We can compress our cost function's two conditional cases into one case:

$$
\mathrm{Cost}(h_\theta(x),y) = - y \; \log(h_\theta(x)) - (1 - y) \log(1 - h_\theta(x))
$$

Notice that when y is equal to 1, then the second term ($(1-y)\log(1-h_\theta(x))$) will be zero and will not affect the result. If y is equal to 0, then the first term ($-y \log(h_\theta(x))$) will be zero and will not affect the result.

We can fully write out our entire cost function as follows:

$$
J(\theta) = - \frac{1}{m} \displaystyle \sum_{i=1}^m [y^{(i)}\log (h_\theta (x^{(i)})) + (1 - y^{(i)})\log (1 - h_\theta(x^{(i)}))]
$$

A vectorized implementation is:

$$
\begin{align*}
& h = g(X\theta)\newline
& J(\theta)  = \frac{1}{m} \cdot \left(-y^{T}\log(h)-(1-y)^{T}\log(1-h)\right)
\end{align*}
$$

**Attention: You should notice that y=1 or y=0 is what we have defined well and we can just use them to compare with our $-log(h_\theta(x))$, so that we can get the cost value. And at the same, you should never ignore that there are m+1 samples in this cost function**

###Gradient Descent

Remember that the general form of gradient descent is:

$$
\begin{align*}
& Repeat \; \lbrace \newline
& \; \theta_j := \theta_j - \alpha \dfrac{\partial}{\partial \theta_j}J(\theta) \newline
& \rbrace
\end{align*}
$$

We can work out the derivative part using calculus to get:

$$
\begin{align*}
& Repeat \; \lbrace \newline
& \; \theta_j := \theta_j - \frac{\alpha}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)} \newline & \rbrace
\end{align*}
$$

Notice that this algorithm is identical to the one we used in linear regression. We still have to simultaneously update all values in theta.

A vectorized implementation is:

$$
\theta := \theta - \frac{\alpha}{m} X^{T} (g(X \theta ) - \vec{y})
$$

![Different linear and logisit regression](../SRC/ml_ch3_3.png)

Attention: **Even though the update rule looks cosmetically identical, because the definition of the hypothesis has changed, this is actually not the same thing as gradient descent for linear regression, what has changed is that the definition for this hypothesis has changed.** Besides, we can monitor a gradient descent like monitoring a linear regression.

###Partial derivative of  $J(\theta)$

Attention: When we work on the optimization algorithm, for gradient descent, we do not actually need code to compute the cost function j of theta. You only need code to compute the derivative terms, but if you think your code as also monitoring convergence then you can compute them all.

First calculate derivative of sigmoid function (it will be useful while finding partial derivative of $J(\theta)$):

$$
\begin{align*}
\sigma(x)'
&=\left(\frac{1}{1+e^{-x}}\right)'
=\frac{0-(-x)'(e^{-x})}{(1+e^{-x})^2}
=\frac{e^{-x}}{(1+e^{-x})^2} \newline
&=\left(\frac{1}{1+e^{-x}}\right)\left(\frac{e^{-x}}{1+e^{-x}}\right)
=\sigma(x)\left(\frac{+1-1 + e^{-x}}{1+e^{-x}}\right)
=\sigma(x)\left(\frac{1 + e^{-x}}{1+e^{-x}} - \frac{1}{1+e^{-x}}\right)
=\sigma(x)(1 - \sigma(x))
\end{align*}
$$

Now we are ready to find out resulting partial derivative:

$$
\begin{align*}
\frac{\partial}{\partial \theta_j} J(\theta) &= 
\frac{\partial}{\partial \theta_j} \frac{-1}{m}\sum_{i=1}^m \left [ y^{(i)} log (h_\theta(x^{(i)})) + (1-y^{(i)}) log (1 - h_\theta(x^{(i)})) \right ] \newline
&= - \frac{1}{m}\sum_{i=1}^m \left [ 
    y^{(i)} \frac{\partial}{\partial \theta_j} log (h_\theta(x^{(i)})) 
  + (1-y^{(i)}) \frac{\partial}{\partial \theta_j} log (1 - h_\theta(x^{(i)}))
\right ] \newline
&= - \frac{1}{m}\sum_{i=1}^m \left [ 
    \frac{y^{(i)} \frac{\partial}{\partial \theta_j} h_\theta(x^{(i)})}{h_\theta(x^{(i)})} 
  + \frac{(1-y^{(i)})\frac{\partial}{\partial \theta_j} (1 - h_\theta(x^{(i)}))}{1 - h_\theta(x^{(i)})}
\right ] \newline
&= - \frac{1}{m}\sum_{i=1}^m \left [ 
    \frac{y^{(i)} \frac{\partial}{\partial \theta_j} \sigma(\theta^T x^{(i)})}{h_\theta(x^{(i)})} 
  + \frac{(1-y^{(i)})\frac{\partial}{\partial \theta_j} (1 - \sigma(\theta^T x^{(i)}))}{1 - h_\theta(x^{(i)})}
\right ] \newline
&= - \frac{1}{m}\sum_{i=1}^m \left [ 
    \frac{y^{(i)} \sigma(\theta^T x^{(i)}) (1 - \sigma(\theta^T x^{(i)})) \frac{\partial}{\partial \theta_j} \theta^T x^{(i)}}{h_\theta(x^{(i)})} 
  + \frac{- (1-y^{(i)}) \sigma(\theta^T x^{(i)}) (1 - \sigma(\theta^T x^{(i)})) \frac{\partial}{\partial \theta_j} \theta^T x^{(i)}}{1 - h_\theta(x^{(i)})}
\right ] \newline
&= - \frac{1}{m}\sum_{i=1}^m \left [ 
    \frac{y^{(i)} h_\theta(x^{(i)}) (1 - h_\theta(x^{(i)})) \frac{\partial}{\partial \theta_j} \theta^T x^{(i)}}{h_\theta(x^{(i)})} 
  - \frac{(1-y^{(i)}) h_\theta(x^{(i)}) (1 - h_\theta(x^{(i)})) \frac{\partial}{\partial \theta_j} \theta^T x^{(i)}}{1 - h_\theta(x^{(i)})}
\right ] \newline
&= - \frac{1}{m}\sum_{i=1}^m \left [ 
    y^{(i)} (1 - h_\theta(x^{(i)})) x^{(i)}_j - (1-y^{(i)}) h_\theta(x^{(i)}) x^{(i)}_j
\right ] \newline
&= - \frac{1}{m}\sum_{i=1}^m \left [ 
    y^{(i)} (1 - h_\theta(x^{(i)})) - (1-y^{(i)}) h_\theta(x^{(i)}) 
\right ] x^{(i)}_j \newline
&= - \frac{1}{m}\sum_{i=1}^m \left [ 
    y^{(i)} - y^{(i)} h_\theta(x^{(i)}) - h_\theta(x^{(i)}) + y^{(i)} h_\theta(x^{(i)}) 
\right ] x^{(i)}_j \newline
&= - \frac{1}{m}\sum_{i=1}^m \left [ y^{(i)} - h_\theta(x^{(i)}) \right ] x^{(i)}_j  \newline
&= \frac{1}{m}\sum_{i=1}^m \left [ h_\theta(x^{(i)}) - y^{(i)} \right ] x^{(i)}_j
\end{align*}
$$

The vectorized version;

$$ 
\nabla J(\theta) = \frac{1}{m} \cdot  X^T \cdot \left(g\left(X\cdot\theta\right) - \vec{y}\right)
$$ 

###Advanced Optimization

"Conjugate gradient", "BFGS", and "L-BFGS" are more sophisticated, faster ways to optimize $\theta$ that can be used instead of gradient descent. A. Ng suggests not to write these more sophisticated algorithms yourself (unless you are an expert in numerical computing) but use the libraries instead, as they're already tested and highly optimized. Octave provides them.

We first need to provide a function that evaluates the following two functions for a given input value $\theta$:

$$
\begin{align*} & J(\theta) \newline & \dfrac{\partial}{\partial \theta_j}J(\theta)\end{align*}
$$

We can write a single function that returns both of these:

    function [jVal, gradient] = costFunction(theta)
      jVal = [...code to compute J(theta)...];
      gradient = [...code to compute derivative of J(theta)...];
    end

```
#Example in the video
function [jVal, gradient] = costFunction(theta)
    jVal = (theta(1)-5)^2 + (theta(2)-5)^2;
    gradient = zeros(2,1);
    gradient(1) = 2*(theta(1)-5);
    gradient(2) = 2*(theta(2)-5);
```

Then we can use octave's "fminunc()" optimization algorithm along with the "optimset()" function that creates an object containing the options we want to send to "fminunc()".

    options = optimset('GradObj', 'on', 'MaxIter', 100);
    initialTheta = zeros(2,1);
    [optTheta, functionVal, exitFlag] = fminunc(@costFunction, initialTheta, options);

We give to the function "fminunc()" our cost function, our initial vector of theta values, and the "options" object that we created beforehand. **Note that $\theta$ must at least a two-dimensional vector.** Read documentation if you need one dimensional function to optimize.

```
#Example in the video
octave:1> options = optimset('GradObj', 'on', 'MaxIter', '100')
options =
  scalar structure containing the fields:
    GradObj = on
    MaxIter = 100
octave:2> initialTheta = zeros(2,1)
initialTheta =
   0
   0
octave:3> [optTheta, functionVal, exitFlag]=fminunc(@costFunction, initialTheta, options)
optTheta =
   5.0000
   5.0000
functionVal =    1.5777e-30
exitFlag =  1
```

<img src="../SRC/ml_ch3_4.png" width="70%">

###Multiclass Classification: One-vs-all

Now we will approach the classification of data into more than two categories. Instead of y = {0,1} we will expand our definition so that y = {0,1...n}.

In this case we divide our problem into $n+1$ ($+1$ because the index starts at 0) binary classification problems, while if we have a multi-class classification problem with k classes y={1,2,3...k}, using the 1-vs-all method, we need k different logistic regression classifiers; in each one, we predict the probability that 'y' is a member of one of our classes.

$$
\begin{align*}
& y \in \lbrace0, 1 ... n\rbrace \newline
& h_\theta^{(0)}(x) = P(y = 0 | x ; \theta) \newline
& h_\theta^{(1)}(x) = P(y = 1 | x ; \theta) \newline
& \cdots \newline
& h_\theta^{(n)}(x) = P(y = n | x ; \theta) \newline
& \mathrm{prediction} = \max_i( h_\theta ^{(i)}(x) )\newline
\end{align*}
$$

We are basically choosing one class and then lumping all the others into a single second class. We do this repeatedly, applying binary logistic regression to each case, and then use the hypothesis that returned the highest value as our prediction.


##Regularization
###The Problem of Overfitting

**Regularization is designed to address the problem of overfitting.**

**High bias** or **underfitting** is when the form of our hypothesis function $h$ maps poorly to the trend of the data. It is usually caused by a function that is too simple or uses too few features.

eg. if we take $h_\theta(x) = \theta_0 + \theta_1x_1 + \theta_2x_2$ then we are making an initial assumption that a linear model will fit the training data well and will be able to generalize but that may not be the case.

At the other extreme, **overfitting** or **high variance** is caused by a hypothesis function that fits the available data but does not generalize well to predict new data. It is usually caused by a complicated function that creates a lot of unnecessary curves and angles unrelated to the data.

This terminology is applied to both linear and logistic regression.

There are **two** main options to address the issue of overfitting:

1.  Reduce the number of features.
    * Manually select which features to keep.
    * Use a model selection algorithm (studied later in the course).
2.  Regularization
    * Keep all the features, but reduce the parameters $\theta_j$.

Regularization works well when we have a lot of slightly useful features.

###Cost Function

If we have overfitting from our hypothesis function, we can reduce the weight that some of the terms in our function carry by increasing their cost.

Say we wanted to make the following function more quadratic:

$$
\theta_0 + \theta_1x + \theta_2x^2 + \theta_3x^3 + \theta_4x^4
$$

We'll want to eliminate the influence of $\theta_3x^3$ and $\theta_4x^4$. Without actually getting rid of these features or changing the form of our hypothesis, we can instead modify our **cost function**:

$$
min_\theta\ \dfrac{1}{2m}\sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2 + 1000\cdot\theta_3^2 + 1000\cdot\theta_4^2
$$

We've added two extra terms at the end to inflate the cost of $\theta_3$ and $\theta_4$. Now, in order for the cost function to get close to zero, we will have to reduce the values of $\theta_3$ and $\theta_4$ to near zero. This will in turn greatly reduce the values of $\theta_3x^3$ and $\theta_4x^4$ in our hypothesis function.

We could also regularize all of our theta parameters in a single summation:

$$
min_\theta\ \dfrac{1}{2m}\ \left[ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2 + \lambda\ \sum_{j=1}^n \theta_j^2 \right]
$$

The $\lambda$, or lambda, is the **regularization parameter**. It determines how much the costs of our theta parameters are inflated. You can visualize the effect of regularization in this interactive plot <https://www.desmos.com/calculator/1hexc8ntqp>

Using the above cost function with the extra summation, we can smooth the output of our hypothesis function to reduce overfitting. If lambda is chosen to be too large, it may smooth out the function too much and cause underfitting.

###Regularized Linear Regression
We can apply regularization to both linear regression and logistic regression. We will approach linear regression first.

####Gradient Descent
We will modify our gradient descent function to separate out $\theta_0$ from the rest of the parameters because we do not want to penalize $\theta_0$.

$$
\begin{align*}
& \text{Repeat}\ \lbrace \newline
& \ \ \ \ \theta_0 := \theta_0 - \alpha\ \frac{1}{m}\ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_0^{(i)} \newline
& \ \ \ \ \theta_j := \theta_j - \alpha\ \left[ \left( \frac{1}{m}\ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)} \right) + \frac{\lambda}{m}\theta_j \right] &\ \ \ \ \ \ \ \ \ \ j \in \lbrace 1,2...n\rbrace\newline
& \rbrace
\end{align*}
$$

The term $\frac{\lambda}{m}\theta_j$ performs our regularization.

With some manipulation our update rule can also be represented as:

$$
\theta_j := \theta_j(1 - \alpha\frac{\lambda}{m}) - \alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}
$$

The first term in the above equation, $1 - \alpha\frac{\lambda}{m}$ will always be less than 1. Intuitively you can see it as reducing the value of $\theta_j$ by some amount on every update.

Notice that the second term is now exactly the same as it was before.

####Normal Equation
Now let's approach regularization using the alternate method of the **non-iterative** normal equation.

To add in regularization, the equation is the same as our original, except that we add another term inside the parentheses:

$$
\begin{align*}
& \theta = \left( X^TX + \lambda \cdot L \right)^{-1} X^Ty \newline
& \text{where}\ \ L = 
\begin{bmatrix}
 0 & & & & \newline
 & 1 & & & \newline
 & & 1 & & \newline
 & & & \ddots & \newline
 & & & & 1 \newline
\end{bmatrix}
\end{align*}
$$

$L$ is a matrix with 0 at the top left and 1's down the diagonal, with 0's everywhere else. It should have dimension $(n + 1) \times (n+1)$. Intuitively, this is the identity matrix (though we are not including $x_0$), multiplied with a single real number $\lambda$.

**Recall that if $m \leq n$, then $X^TX$ is non-invertible. However, when we add the term $\lambda\cdot L$, then $X^TX$ + $\lambda\cdot L$ becomes invertible.**

###Regularized Logistic Regression

We can regularize logistic regression in a similar way that we regularize linear regression. Let's start with the cost function.

####Cost Function

Recall that our cost function for logistic regression was:

$$
J(\theta) = - \frac{1}{m} \sum_{i=1}^m \large[ y^{(i)}\ \log (h_\theta (x^{(i)})) + (1 - y^{(i)})\ \log (1 - h_\theta(x^{(i)})) \large]
$$

We can regularize this equation by adding a term to the end:

$$
J(\theta) = - \frac{1}{m} \sum_{i=1}^m \large[ y^{(i)}\ \log (h_\theta (x^{(i)})) + (1 - y^{(i)})\ \log (1 - h_\theta(x^{(i)}))\large] + \frac{\lambda}{2m}\sum_{j=1}^n \theta_j^2
$$

**Note Well:** The second sum, $\sum_{j=1}^n \theta_j^2$ __means to explicitly exclude__ the bias term, $\theta_{0}$. I.e. the $\theta$ vector is indexed from $0$ to $n$ (holding $n+1$ values, $\theta_{0}$ through $\theta_{n}$), and this sum explicitly skips $\theta_{0}$, by running from $1$ to $n$, skipping $0$.

####Gradient Descent

Just like with linear regression, we will want to **separately** update $\theta_0$ and the rest of the parameters because we do not want to regularize $\theta_0$.

$$
\begin{align*}
& \text{Repeat}\ \lbrace \newline
& \ \ \ \ \theta_0 := \theta_0 - \alpha\ \frac{1}{m}\ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_0^{(i)} \newline
& \ \ \ \ \theta_j := \theta_j - \alpha\ \left[ \left( \frac{1}{m}\ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)} \right) + \frac{\lambda}{m}\theta_j \right] &\ \ \ \ \ \ \ \ \ \ j \in \lbrace 1,2...n\rbrace\newline
& \rbrace
\end{align*}
$$

This is identical to the gradient descent function presented for linear regression.
















[Lecture 6](../SRC/ml_lecture6.pdf)
[Lecture 7](../SRC/ml_lecture7.pdf)




























































