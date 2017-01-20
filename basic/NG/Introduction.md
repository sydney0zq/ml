[TOC]

##What is machine learning?
Two definitions of Machine Learning are offered. Arthur Samuel described it as: "the field of study that gives computers the ability to learn without being explicitly programmed." This is an older, informal definition.

Tom Mitchell provides a more modern definition: "A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E."

In general, any machine learning problem can be assigned to one of two broad classifications:
- supervised learning, OR
- unsupervised learning.

###Supervised Learning
In supervised learning, we are given a data set and already know what our correct output should look like, having the idea that there is a relationship between the input and the output.

Supervised learning problems are categorized into "**regression**" and "**classification**" problems. In a **regression** problem, we are trying to predict results within a **continuous** output, meaning that we are trying to map input variables to some **continuous** function. In a **classification** problem, we are instead trying to predict results in a **discrete** output. In other words, we are trying to map input variables into **discrete** categories. 

> Example 1:
Given data about the size of houses on the real estate market, try to predict their price. Price as a function of size is a continuous output, so this is a **regression** problem.
We could turn this example into a classification problem by instead making our output about whether the house "sells for more or less than the asking price." Here we are classifying the houses based on price into two **discrete** categories.
>
> Example 2: (a)Regression - Given a picture of Male/Female, We have to predict his/her age on the basis of given picture. (b)Classification - Given a picture of Male/Female, We have to predict Whether He/She is of High school, College, Graduate age. Another Example for Classification - Banks have to decide whether or not to give a loan to someone on the basis of his credit history.

###Unsupervised Learning
Unsupervised learning, on the other hand, allows us to approach problems with little or no idea what our results should look like. We can derive structure from data where we don't necessarily know the effect of the variables.

We can derive this structure by **clustering** the data based on relationships among the variables in the data.

With unsupervised learning there is no feedback based on the prediction results, i.e., there is no teacher to correct you.

> Example: 
> 
> *Clustering*:  Take a collection of 1000 essays written on the US Economy, and find a way to automatically group these essays into a small number that are somehow similar or related by different variables, such as word frequency, sentence length, page count, and so on.
>
>*Non-clustering*:  The "Cocktail Party Algorithm", which can find structure in messy data. [wiki](https://www.wikiwand.com/en/Cocktail_party_effect) An answer on Quora to enhance understanding : [What is the difference between supervised and unsupervised learning algorithms?--Quora](https://www.quora.com/What-is-the-difference-between-supervised-and-unsupervised-learning-algorithms)


##Linear Regression with One Variable
###Model Representation
Recall that in *regression problems*, we are taking input variables and trying to fit the output onto a *continuous* expected result function.
Linear regression with one variable is also known as "univariate linear regression."

Univariate linear regression is used when you want to predict a **single output** value $y$ from a **single input** value $x$. We're doing **supervised learning** here, so that means we already have an idea about what the input/output cause and effect should be.

###The Hypothesis Function
Our hypothesis function has the general form:
$$\hat {y} = h{_\theta}(x)=\theta_0 + \theta{_1}x$$

Note that this is like the equation of a straight line. We give to $h_\theta(x)$ values for $\theta_0$ and $\theta_1$ to get our estimated output $\hat{y}$. In other words, we are trying to create a function called $h_\theta$ that is trying to map our input data (the x's) to our output data (the y's).

> **Example**
>
>Suppose we have the following set of training data:
```
input output
x           y
0           4
1           7
2           7
3           8
```
Now we can make a random guess about our $h_\theta$ function: $\theta_0=2$ and $\theta_1=2$. The hypothesis function becomes $h_\theta(x)=2+2x$.

###Cost Function
We can measure the accuracy of our hypothesis function by using a **cost function**. This takes an average (actually a fancier version of an average) of all the results of the hypothesis with inputs from x's compared to the actual output y's.
$$J(\theta_0,\theta_1)=\frac{1}{2m}\sum_{i=1}^{m}{(\hat y_i-y_i)^2}=\frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x_i)-y_i)^2$$

To break it apart, it is $\frac {1}{2} \bar x$ where $\bar x$ is the mean of the squares of $h_\theta(x_i)-y_i$ , or the difference between the predicted value and the actual value.

This function is otherwise called the "Squared error function", or "Mean squared error". The mean is halved $\frac{1}{2m}$ as a convenience for the computation of the gradient descent, as the derivative term of the square function will cancel out the $\frac{1}{2}$ term.

Now we are able to concretely measure the accuracy of our predictor function against the correct results we have so that we can predict new results we don't have.

If we try to think of it in visual terms, our training data set is scattered on the x-y plane. We are trying to make straight line (defined by $h_\theta(x)$ which passes through this scattered set of data. Our objective is to get the best possible line. The best possible line will be such so that the average squared vertical distances of the scattered points from the line will be the least. In the best case, the line should pass through all the points of our training data set. In such a case the value of $J(\theta_0,\theta_1)$will be 0.

###Frequently Asked Questions
**Q: Why is the cost function about the sum of squares, rather than the sum of cubes (or just the (h(x)−y) or abs(h(x)−y) ) ?**

A: It might be easier to think of this as measuring the distance of two points. In this case, we are measuring the distance of two multi-dimensional values (i.e. the observed output value $y_i$and the estimated output value $\hat y_i$). We all know how to measure the distance of two points $(x_1,y_1)$ and $(x_2,y_2)$, which is $\sqrt { (x_1-x_2)^2 + (y_1-y_2)^2 }$. If we have n-dimension then we want the positive square root of $\sum_{i=1}^n(x_i-y_i)^2$ That's where the sum of squares comes from. 

The sum of squares isn’t the only possible cost function, but **it has many nice properties**. Squaring the error means that an overestimate is "punished" just the same as an underestimate: an error of -1 is treated just like +1, and the two equal but opposite errors can’t cancel each other. If we cube the error (or just use the difference), we lose this property. Also in the case of cubing, big errors are punished more than small ones, so an error of 2 becomes 8.

**The squaring function is smooth (can be differentiated) and yields linear forms after differentiation, which is nice for optimization. It also has the property of being “convex”. A convex cost function guarantees there will be a global minimum, so our algorithms will converge.**

**If you throw in absolute value, then you get a non-differentiable function. If you try to take the derivative of abs(x) and set it equal to zero to find the minimum, you won't get any answers since it's undefined in 0.**
<hr>

**Q: Why can’t I use 4th powers in the cost function? Don’t they have the nice properties of squares?**

A: Imagine that you are throwing darts at a dartboard, or firing arrows at a target.If you use the sum of squares as the error (where the center of the bulls-eye is the origin of the coordinate system), the error is the distance from the center. Now rotate the coordinates by 30 degree, or 45 degrees, or anything. The distance, and hence the error, remains unchanged.  4th powers lack this property, which is known as “**rotational invariance**”.
<hr>
**Q: Why does 1/(2 * m) make the math easier?**

A: When we differentiate the cost to calculate the gradient, we get a factor of 2 in the numerator, due to the exponent inside the sum. This '2' in the numerator cancels-out with the '2' in the denominator, saving us one math operation in the formula.

##Gradient Descent
So we have our hypothesis function and we have a way of measuring how well it fits into the data. Now we need to **estimate the parameters in hypothesis function**. That's where gradient descent comes in.

Imagine that we graph our hypothesis function based on its fields $\theta_0$ and $\theta_1$ (actually we are graphing the cost function as a function of the parameter estimates). This can be kind of confusing; we are moving up to a higher level of abstraction. We are not graphing $x$ and $y$ itself, but the parameter range of our hypothesis function and the cost resulting from selecting particular set of parameters.

We put $\theta_0$ on the x axis and $\theta_1$ on the y axis, with the cost function on the vertical z axis. The points on our graph will be the result of the **cost function** using our hypothesis with those specific theta parameters.

We will know that we have succeeded when our cost function is at the very bottom of the pits in our graph, i.e. when its value is the minimum.

The way we do this is by taking the **derivative** (the tangential line to a function) of our cost function. The slope of the tangent is the derivative at that point and it will give us a direction to move towards. We make steps down the cost function in the direction with the steepest descent, and the size of each step is determined by the parameter $\alpha$, which is called the **learning rate**.

The gradient descent algorithm is:

repeat until  convergence:
$$\theta_j := \theta_j - \alpha \frac{\partial }{\partial \theta_j}J(\theta_0,\theta_1)$$
where $j=0,1$ represents the feature index number.

>Correct: Simultaneous update
>$temp0 := \theta_0-\alpha\frac{\partial}{\partial\theta_0}J(\theta_0,\theta_1)$
>$temp1 := \theta_1-\alpha\frac{\partial}{\partial\theta_1}J(\theta_0,\theta_1)$
>$\theta_0 := temp0$
>$\theta_1 := temp1$
>
>Incorrect:
>$temp0 := \theta_0-\alpha\frac{\partial}{\partial\theta_0}J(\theta_0,\theta_1)$
>$\theta_0 := temp0$
>$temp1 := \theta_1-\alpha\frac{\partial}{\partial\theta_1}J(\theta_0,\theta_1)$
>$\theta_1 := temp1$

Intuitively, this could be thought of as:

repeat until convergence:
$$\theta_j := \theta_j-\alpha$$[Slope of tangent aka derivative in j dimension]

###Gradient Descent for Linear Regression
When specifically applied to the case of linear regression, a new form of the gradient descent equation can be derived. We can substitute our actual cost function and our actual hypothesis function and modify the equation to ([More](https://math.stackexchange.com/questions/70728/partial-derivative-in-gradient-descent-for-two-variables/189792#189792)):
repeat until convergence: {
$$\begin{align}
\theta_0 := \theta_0 - \alpha \frac{1}{m}\sum_{i=1}^{m}(h_\theta(x_i)-y_i)\newline
\theta_1 := \theta_1 - \alpha \frac{1}{m}\sum_{i=1}^{m}((h_\theta(x_i)-y_i)x_i)\newline
\text{You should clearly know that:} \newline
\frac{1}{m}\sum_{i=1}^{m}(h_\theta(x_i)-y_i)=\frac {\partial}{\partial\theta_0} J(\theta_0, \theta_1)\newline
\frac{1}{m}\sum_{i=1}^{m}((h_\theta(x_i)-y_i)x_i)=\frac {\partial}{\partial\theta_1} J(\theta_0, \theta_1)\newline
\end{align}$$
}
where $m$ is the size of the training set, $\theta_0$ a constant that will be changing simultaneously with $\theta_1$ and $x_i, y_i$ are values of the given training set (data), $\alpha$ is learning rate, which controls how big a step we take when updating parameter $\theta _J$. **Notice that $\alpha$ is always positive!**

If $\alpha$ is too small, gradient descent can be slow.
If $\alpha$ is too large, gradient descent can overshoot the minimum.  It may fail to converge, or even diverge.

**Gradient descent can converge to a local minium, even with the learning rate $\alpha$ fixed.** Because as we approach a local minimum, gradient descent will automatically take smaller steps. So no need to decrease $\alpha$ over time.

Note that we have separated out the two cases for $\theta_j$ into separate equations for $\theta_0$ and $\theta_1$; and that for $\theta_1$ we are multiplying $x_i$ at the end due to the derivative.

The point of all this is that if we start with a guess for our hypothesis and then repeatedly apply these gradient descent equations, our hypothesis will become more and more accurate.

###Gradient Descent for Linear Regression: visual worked example
[Youtube Linear Regression](https://www.youtube.com/watch?v=WnqQrPNYz5Q)


##Linear Algebra Review
###Matrices and Vectors
Matrices are 2-dimensional arrays:
$$
 \begin{bmatrix}
  a & b & c \newline 
  d & e & f \newline 
  g & h & i \newline 
  j & k & l
 \end{bmatrix}
$$

**Notation and terms**

-  $A_{ij}$ refers to the element in the ith row and jth column of matrix A.
*  A vector with 'n' rows is referred to as an **'n'-dimensional **vector
*  $v_i$ refers to the element in the ith row of the vector.
*  In general, all our vectors and matrices will be 1-indexed. Note that for some programming languages, the arrays are 0-indexed.
*  Matrices are usually denoted by uppercase names while vectors are lowercase.
*  "Scalar" means that an object is a single value, not a vector or matrix. 
*  $\mathbb{R}$ refers to the set of scalar real numbers
*  $\mathbb{R^n}$ refers to the set of n-dimensional vectors of real numbers

###Matrix-Vector Multiplication
We map the column of the vector onto each row of the matrix, multiplying each element and summing the result.
$$
 \begin{bmatrix}
  a & b \newline 
  c & d \newline 
  e & f
 \end{bmatrix} *
\begin{bmatrix}
  x \newline 
  y \newline 
 \end{bmatrix} =
\begin{bmatrix}
  a*x + b*y \newline 
  c*x + d*y \newline 
  e*x + f*y
 \end{bmatrix}
$$
The result is a **vector**. The vector must be the **second** term of the multiplication. The number of **columns** of the matrix must equal the number of **rows** of the vector.

An **m x n matrix** multiplied by an **n x 1 vector** results in an **m x 1 vector**.

###Matrix-Matrix Multiplication
We multiply two matrices by breaking it into several vector multiplications and concatenating the result
$$
 \begin{bmatrix}
  a & b \newline 
  c & d \newline 
  e & f
 \end{bmatrix} *
\begin{bmatrix}
  w & x \newline 
  y & z \newline 
 \end{bmatrix} =
\begin{bmatrix}
  a*w + b*y & a*x + b*z \newline 
  c*w + d*y & c*x + d*z \newline 
  e*w + f*y & e*x + f*z
 \end{bmatrix}
$$
An **m x n matrix** multiplied by an **n x o matrix** results in an **m x o** matrix. In the above example, a 3 x 2 matrix times a 2 x 2 matrix resulted in a 3 x 2 matrix.

To multiply two matrices, the number of **columns** of the first matrix must equal the number of **rows** of the second matrix.

###Matrix Multiplication Properties

* Not commutative. $A*B \neq B*A$
* Associative. $(A*B)*C = A*(B*C)$

The **identity matrix**, when multiplied by any matrix of the same dimensions, results in the original matrix. It's just like multiplying numbers by 1. The identity matrix simply has 1's on the diagonal (upper left to lower right diagonal) and 0's elsewhere.
$$
 \begin{bmatrix}
  1 & 0 & 0 \newline 
  0 & 1 & 0 \newline 
  0 & 0 & 1 \newline 
 \end{bmatrix}
$$

###Inverse and Transpose
The **inverse** of a matrix $A$ is denoted $A^{-1}$. Multiplying by the inverse results in the identity matrix.

A non square matrix does not have an inverse matrix. We can compute inverses of matrices in octave with the pinv(A) function $^{[1]}$ and in matlab with the inv(A) function. Matrices that don't have an inverse are **singular** or **degenerate**.

$$
A = 
 \begin{bmatrix}
  a & b \newline 
  c & d \newline 
  e & f
 \end{bmatrix} 
$$

$$
A^T = 
 \begin{bmatrix}
  a & c & e \newline 
  b & d & f \newline 
 \end{bmatrix}
$$

In other words:
$$A_{ij} = A^T_{ji}$$

























