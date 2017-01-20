[TOC]

##Linear Regression with Multiple Variables
###Multiple Features
Linear regression with multiple variables is also known as "multivariate linear regression".
We now introduce notation for equations where we can have any number of input variables.
$$
\begin{align*}
x_j^{(i)} &= \text{value of feature } j \text{ in the }i^{th}\text{ training example} \newline
x^{(i)}& = \text{the column vector of all the feature inputs of the }i^{th}\text{ training example} \newline
m &= \text{the number of training examples} \newline
n &= \left| x^{(i)} \right| ; \text{(the number of features)} 
\end{align*}
$$
Now define the multivariable form of the hypothesis function as follows, accomodating these multiple features:
$$
h_\theta (x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_3 + \cdots + \theta_n x_n
$$
In order to develop intuition about this function, we can think about $\theta_0$ as the basic price of a house, $\theta_1$ as the price per square meter, $\theta_2$ as the price per floor, etc.
$x_1$ will be the number of square meters in the house, $x_2$ the number of floors, etc.

Using the definition of matrix multiplication, our multivariable hypothesis function can be concisely represented as:
$$
\begin{align*}
h_\theta(x) =
\begin{bmatrix}
\theta_0 \hspace{2em}  \theta_1 \hspace{2em}  ...  \hspace{2em}  \theta_n
\end{bmatrix}
\begin{bmatrix}
x_0 \newline
x_1 \newline
\vdots \newline
x_n
\end{bmatrix}
= \theta^T x
\end{align*}
$$

Remark: Note that for convenience reasons in this course Mr. Ng assumes $x_{0}^{(i)}  =1 \text{ for }  (i\in \{ 1,\dots, m \} )$

[**Note**: So that we can do matrix operations with theta and x, we will set $x^{(i)}_0 = 1$, for all values of $i$. This makes the two vectors 'theta' and $x^{(i)}$ match each other element-wise (that is, have the same number of elements: $n + 1$).]


The training examples are stored in $X$ row-wise, like such:

$$
\begin{align*}
X = 
\begin{bmatrix}
x^{(1)}_0 & x^{(1)}_1  \newline
x^{(2)}_0 & x^{(2)}_1  \newline
x^{(3)}_0 & x^{(3)}_1  
\end{bmatrix}
&
,\theta = 
\begin{bmatrix}
\theta_0 \newline
\theta_1 \newline
\end{bmatrix}
\end{align*}
$$

You can calculate the hypothesis as a column vector of size (m x 1) with:
$$
h_\theta(X) = X \theta
$$

###Cost function

For the parameter vector $\theta$ (of type $\mathbb{R}^{n+1}$ or in $\mathbb{R}^{(n+1) \times 1}$), the cost function is:

$$J(\theta) = \dfrac {1}{2m} \displaystyle \sum_{i=1}^m \left (h_\theta (x^{(i)}) - y^{(i)} \right)^2$$

The vectorized version is:

$$J(\theta) = \dfrac {1}{2m} (X\theta - \vec{y})^{T} (X\theta - \vec{y})$$

Where $\vec{y}$ denotes the vector of all y values.

###Gradient Descent for Multiple Variables

The gradient descent equation itself is generally the same form; we just have to repeat it for our 'n' features:

$$\begin{align*}
& \text{repeat until convergence:} \; \lbrace \newline 
\; & \theta_0 := \theta_0 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_0^{(i)}\newline
\; & \theta_1 := \theta_1 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_1^{(i)} \newline
\; & \theta_2 := \theta_2 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_2^{(i)} \newline
& \cdots
\newline \rbrace
\end{align*}$$

In other words:

$$\begin{align*}
& \text{repeat until convergence:} \; \lbrace \newline 
\; & \theta_j := \theta_j - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)} \;  & \text{for j := 0..n}
\newline \rbrace
\end{align*}$$

![Comparison Of Two Equations](../SRC/ml_ch2_0.png)


###Matrix Notation

The Gradient Descent rule can be expressed as:

$$
\large
\theta := \theta - \alpha \nabla J(\theta)
$$


Where $\nabla J(\theta)$ is a column vector of the form:

$$\large
\nabla J(\theta)  = \begin{bmatrix}
\frac{\partial J(\theta)}{\partial \theta_0}   \newline
\frac{\partial J(\theta)}{\partial \theta_1}   \newline
\vdots   \newline
\frac{\partial J(\theta)}{\partial \theta_n} 
\end{bmatrix}
$$

The j-th component of the gradient is the summation of the product of two terms:

$$\begin{align*}
\; &\frac{\partial J(\theta)}{\partial \theta_j} &=&  \frac{1}{m} \sum\limits_{i=1}^{m}  \left(h_\theta(x^{(i)}) - y^{(i)} \right) \cdot x_j^{(i)} \newline
\; & &=& \frac{1}{m} \sum\limits_{i=1}^{m}   x_j^{(i)} \cdot \left(h_\theta(x^{(i)}) - y^{(i)}  \right) 
\end{align*}$$


Sometimes, the summation of the product of two terms can be expressed as the product of two vectors. 

Here, $x_j^{(i)}$, for $i=1,...,m$, represents the $m$ elements of the $j$-th column, $\vec{x_j}$, of the training set $X$. 

The other term $\left(h_\theta(x^{(i)}) - y^{(i)}  \right)$ is the vector of the deviations between the predictions $h_\theta(x^{(i)})$ and the true values $y^{(i)}$. Re-writing $\frac{\partial J(\theta)}{\partial \theta_j}$, we have:

$$\begin{align*}
\; &\frac{\partial J(\theta)}{\partial \theta_j} &=& \frac1m  \vec{x_j}^{T} (X\theta - \vec{y}) \newline
\newline
\newline
\; &\nabla J(\theta) & = & \frac 1m X^{T} (X\theta - \vec{y}) \newline
\end{align*}$$

Finally, the matrix notation (vectorized) of the Gradient Descent rule is:

 $$
\large
\theta := \theta - \frac{\alpha}{m} X^{T} (X\theta - \vec{y})
$$

####Feature Normalization(Practice I)

We can speed up gradient descent by having each of our input values in roughly the same range. **This is because** $\theta$ **will descend quickly on small ranges and slowly on large ranges, and so will oscillate inefficiently down to the optimum when the variables are very uneven.**

The way to prevent this is to modify the ranges of our input variables so that they are all roughly the same. Ideally:
$$ -1 \le x_i \le 1 $$
or
$$ -0.5 \le x_i \le 0.5 $$

These aren't exact requirements; we are only trying to speed things up.  The goal is to get all input variables into roughly one of these ranges, give or take a few.

Two techniques to help with this are **feature scaling** and **mean normalization**.  Feature scaling involves dividing the input values by the range (i.e. the maximum value minus the minimum value) of the input variable, resulting in a new range of just 1.  Mean normalization involves subtracting the average value for an input variable from the values for that input variable, resulting in a new average value for the input variable of just zero.  To implement both of these techniques, adjust input values as shown in this formula:

$$
x_i := \dfrac{x_i - \mu_i}{s_i}
$$

Where $\mu_i$ is the **average** of all the values for feature (i) and $s_i$ is the range of values (max - min), or $s_i$ is the standard deviation. 
**Note that dividing by the range, or dividing by the standard deviation, give different results.** The quizzes in this course use range - the programming exercises use standard deviation.

Example: $x_i$ is housing prices with range of 100 to 2000, with a mean value of 1000. Then, $x_i := \dfrac{price-1000}{1900}$.

####Gradient Descent Tips

**Debugging gradient descent.** Make a plot with *number of iterations* on the x-axis. Now plot the cost function, $J(\theta)$ over the number of iterations of gradient descent. If $J(\theta)$ ever increases, then you probably need to decrease $\alpha$.

**Automatic convergence test.** Declare convergence if $J(\theta)$ decreases by less than $E$ in one iteration, where $E$ is some small value such as $10^{-3}$. However in practice it's difficult to choose this threshold value.

It has been proven that if learning rate $\alpha$ is sufficiently small, then $J(\theta)$ will decrease on every iteration. Andrew Ng recommends decreasing $\alpha$ by multiples of 3.

![How to debug gradient descent](../SRC/ml_ch2_1.png)
![Check the curve](../SRC/ml_ch2_2.png)

###Features and Polynomial Regression

We can improve our features and the form of our hypothesis function in a couple different ways.

We can **combine** multiple features into one. For example, we can combine $x_1$ and $x_2$ into a new feature $x_3$ by taking $x_1 \cdot x_2$. 

####Polynomial Regression

Our hypothesis function need not be linear (a straight line) if that does not fit the data well.

We can **change the behavior or curve** of our hypothesis function by making it a quadratic, cubic or square root function (or any other form). 

For example, if our hypothesis function is $h_\theta(x) = \theta_0 + \theta_1 x_1$ then we can create additional features based on $x_1$, to get the quadratic function $h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_1^2$ or the cubic function  $h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_1^2 + \theta_3 x_1^3$                                                                               

In the cubic version, we have created new features $x_2$ and $x_3$ where $x_2 = x_1^2$ and $x_3 = x_1^3$.

To make it a square root function, we could do: $h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 \sqrt{x_1}$

The curve that Prof Ng discusses about "doesn't ever come back down" is in reference to the hypothesis function that uses the sqrt() function  not the one that uses ${size}^2$ . The quadratic form of the hypothesis function would have the shape shown with the blue dotted line if $\theta_2$ was negative.    

**One important thing to keep in mind is, if you choose your features this way then feature scaling becomes very important.**

eg. if $x_1$ has range 1 - 1000 then range of $x_1^2$ becomes 1 - 1000000 and that of $x_1^3$ becomes 1 - 1000000000               

###Normal Equation

> My notes: On lecture, Ng uses n to denote the feature number while m to denote the training number.

The "Normal Equation" is a method of finding the optimum theta **without iteration.**

$$
\theta = (X^T X)^{-1}X^T y
$$

There is **no need** to do feature scaling with the normal equation.

Mathematical proof of the Normal equation requires knowledge of linear algebra and is fairly involved, so you do not need to worry about the details.

Proofs:  [Wikipedia](http://en.wikipedia.org/wiki/Linear_least_squares_(mathematics)) [Eli Bendersky](http://eli.thegreenplace.net/2014/derivation-of-the-normal-equation-for-linear-regression)

The following is a comparison of gradient descent and the normal equation:

<table border=1 align="center">
<tr><th>Gradient Descent</th><th>Normal Equation</th></tr>
<tr><td>Need to choose alpha</td><td>No need to choose alpha</td></tr>
<tr><td>Needs many iterations</td><td>No need to iterate</td></tr>
<tr><td>$$O~(kn^2)$$</td><td>$$O~(n^3)$$, need to calculate inverse of $$X^TX$$ </td></tr>
<tr><td>Works well when n is large</td><td>Slow if n is very large</td></tr>
</table>

With the normal equation, computing the inversion has complexity $\mathcal{O}(n^3)$. So if we have a very large number of features, the normal equation will be slow. In practice, when n exceeds 10,000 it might be a good time to go from a normal solution to an iterative process.

####Normal Equation Noninvertibility

When implementing the normal equation in octave we want to use the 'pinv' function rather than 'inv.'

$X^T X$ may be **noninvertible**. The common causes are:                                                                           

* Redundant features, where two features are very closely related (i.e. they are linearly dependent)
* Too many features (e.g. $m \leq n$). In this case, delete some features or use "regularization".     


###ML:Octave Tutorial

####Basic Operations
        %% Change Octave prompt  
	PS1('>> ');
	%% Change working directory in windows example:
	cd 'c:/path/to/desired/directory name'
	%% Note that it uses normal slashes and does not use escape characters for the empty spaces.

	%% elementary operations
	5+6
	3-2
	5*8
	1/2
	2^6
	1 == 2  % false
	1 ~= 2  % true.  note, not "!="
	1 && 0
	1 || 0
	xor(1,0)

	%% variable assignment
	a = 3; % semicolon suppresses output
	b = 'hi';
	c = 3>=1;

	% Displaying them:
	a = pi
	disp(a)
	disp(sprintf('2 decimals: %0.2f', a))
	disp(sprintf('6 decimals: %0.6f', a))
	format long
	a
	format short
	a

	%%  vectors and matrices
	A = [1 2; 3 4; 5 6]

	v = [1 2 3]
	v = [1; 2; 3]
	v = 1:0.1:2    % from 1 to 2, with stepsize of 0.1. Useful for plot axes
	v = 1:6        % from 1 to 6, assumes stepsize of 1 (row vector)

	C = 2*ones(2,3)  % same as C = [2 2 2; 2 2 2]
	w = ones(1,3)    % 1x3 vector of ones
	w = zeros(1,3)
	w = rand(1,3)  % drawn from a uniform distribution 
	w = randn(1,3) % drawn from a normal distribution (mean=0, var=1)
	w = -6 + sqrt(10)*(randn(1,10000));  % (mean = -6, var = 10) - note: add the semicolon
	hist(w)     % plot histogram using 10 bins (default)
	hist(w,50)  % plot histogram using 50 bins
    % note: if hist() crashes, try "graphics_toolkit('gnu_plot')" 

	I = eye(4)    % 4x4 identity matrix

	% help function
	help eye
	help rand
	help help

####Moving Data Around


	%% dimensions
	sz = size(A) % 1x2 matrix: [(number of rows) (number of columns)]
	size(A,1)  % number of rows
	size(A,2)  % number of cols
	length(v)  % size of longest dimension


	%% loading data
	pwd    % show current directory (current path)
	cd 'C:\Users\ang\Octave files'   % change directory 
	ls     % list files in current directory 
	load q1y.dat    % alternatively, load('q1y.dat')
	load q1x.dat
	who    % list variables in workspace
	whos   % list variables in workspace (detailed view) 
	clear q1y       % clear command without any args clears all vars
	v = q1x(1:10);  % first 10 elements of q1x (counts down the columns)
	save hello.mat v;   % save variable v into file hello.mat
	save hello.txt v -ascii; % save as ascii
	% fopen, fread, fprintf, fscanf also work  [[not needed in class]]

	%% indexing
	A(3,2)  % indexing is (row,col)
	A(2,:)  % get the 2nd row. 
			% ":" means every element along that dimension
	A(:,2)  % get the 2nd col
	A([1 3],:) % print all  the elements of rows 1 and 3

	A(:,2) = [10; 11; 12]     % change second column
	A = [A, [100; 101; 102]]; % append column vec
	A(:) % Select all elements as a column vector.

	% Putting data together 
	A = [1 2; 3 4; 5 6]
	B = [11 12; 13 14; 15 16] % same dims as A
	C = [A B]  % concatenating A and B matrices side by side
	C = [A, B] % concatenating A and B matrices side by side
	C = [A; B] % Concatenating A and B top and bottom

####Computing on Data

%% initialize variables
	A = [1 2;3 4;5 6]
	B = [11 12;13 14;15 16]
	C = [1 1;2 2]
	v = [1;2;3]

	%% matrix operations
	A * C  % matrix multiplication
	A .* B % element-wise multiplication
	% A .* C  or A * B gives error - wrong dimensions
	A .^ 2 % element-wise square of each element in A
	1./v   % element-wise reciprocal
	log(v)  % functions like this operate element-wise on vecs or matrices 
	exp(v)
	abs(v)

	-v  % -1*v

	v + ones(length(v), 1)  
	% v + 1  % same

	A'  % matrix transpose

	%% misc useful functions

	% max  (or min)
	a = [1 15 2 0.5]
	val = max(a)
	[val,ind] = max(a) % val -  maximum element of the vector a and index - index value where maximum occur
	val = max(A) % if A is matrix, returns max from each column

	% compare values in a matrix & find
	a < 3 % checks which values in a are less than 3
	find(a < 3) % gives location of elements less than 3
	A = magic(3) % generates a magic matrix - not much used in ML algorithms
	[r,c] = find(A>=7)  % row, column indices for values matching comparison

	% sum, prod
	sum(a)
	prod(a)
	floor(a) % or ceil(a)
	max(rand(3),rand(3))
	max(A,[],1) -  maximum along columns(defaults to columns - max(A,[]))
	max(A,[],2) - maximum along rows
	A = magic(9)
	sum(A,1)
	sum(A,2)
	sum(sum( A .* eye(9) ))
	sum(sum( A .* flipud(eye(9)) ))


	% Matrix inverse (pseudo-inverse)
	pinv(A)        % inv(A'*A)*A'

####Plotting Data

	%% plotting
	t = [0:0.01:0.98];
	y1 = sin(2*pi*4*t); 
	plot(t,y1);
	y2 = cos(2*pi*4*t);
	hold on;  % "hold off" to turn off
	plot(t,y2,'r');
	xlabel('time');
	ylabel('value');
	legend('sin','cos');
	title('my plot');
	print -dpng 'myPlot.png'
	close;           % or,  "close all" to close all figs
    figure(1); plot(t, y1);
    figure(2); plot(t, y2);
	figure(2), clf;  % can specify the figure number
	subplot(1,2,1);  % Divide plot into 1x2 grid, access 1st element
	plot(t,y1);
	subplot(1,2,2);  % Divide plot into 1x2 grid, access 2nd element
	plot(t,y2);
	axis([0.5 1 -1 1]);  % change axis scale

	%% display a matrix (or image) 
	figure;
	imagesc(magic(15)), colorbar, colormap gray;
	% comma-chaining function calls.  
	a=1,b=2,c=3
	a=1;b=2;c=3;

#### Control statements: `for`, `while`, `if` statements

	v = zeros(10,1);
	for i=1:10, 
		v(i) = 2^i;
	end;
	% Can also use "break" and "continue" inside for and while loops to control execution.

	i = 1;
	while i <= 5,
	  v(i) = 100; 
	  i = i+1;
	end

	i = 1;
	while true, 
	  v(i) = 999; 
	  i = i+1;
	  if i == 6,
		break;
	  end;
	end

	if v(1)==1,
	  disp('The value is one!');
	elseif v(1)==2,
	  disp('The value is two!');
	else
	  disp('The value is not one or two!');
	end
	
####Functions

To create a function, type the function code in a text editor (e.g. gedit or notepad), and save the file as "functionName.m" 

Example function:

	function y = squareThisNumber(x)

	y = x^2;

To call the function in Octave, do either:

1) Navigate to the directory of the functionName.m file and call the function:

        % Navigate to directory:
        cd /path/to/function

        % Call the function:
        functionName(args)

2) Add the directory of the function to the load path and save it:<br>
**You should not use addpath/savepath for any of the assignments in this course. Instead use 'cd' to change the current working directory. Watch the video on submitting assignments in week 2 for instructions.** 

        % To add the path for the current session of Octave:
        addpath('/path/to/function/')

        % To remember the path for future sessions of Octave, after executing addpath above, also do:
        savepath

Octave's functions can return more than one value:

        function [y1, y2] = squareandCubeThisNo(x)
        y1 = x^2
        y2 = x^3

Call the above function this way:

        [a,b] = squareandCubeThisNo(x)

####Vectorization

Vectorization is the process of taking code that relies on **loops** and converting it into **matrix operations**. It is more efficient, more elegant, and more concise.

As an example, let's compute our prediction from a hypothesis. Theta is the vector of fields for the hypothesis and x is a vector of variables.

With loops:

	prediction = 0.0;
	for j = 1:n+1,
	  prediction += theta(j) * x(j);
	end;

With vectorization:

	prediction = theta' * x;

If you recall the definition multiplying vectors, you'll see that this one operation does the element-wise multiplication and overall sum in a very concise notation.

####External Resources

[Octave Quick Reference](../SRC/refcard-a4.pdf)

[An Introduction to Matlab](http://www.maths.dundee.ac.uk/ftp/na-reports/MatlabNotes.pdf)

[Learn X in Y Minutes: Matlab](https://learnxinyminutes.com/docs/matlab/)

**Q: Where is the MATLAB tutorial?**

A: Octave and MATLAB are mostly identical for the purposes of this course. The differences are minor and and are pointed-out in the lecture notes in the Wiki, and in the Tutorials for the programming exercises (see the Forum for a list of Tutorials).











TODO: 就像是内积一样, 结合图书馆的向量内积来看。(找到那本线性代数的书)
[网易云音乐推荐算法](https://www.zhihu.com/question/26743347)

多项式正交性?或者相互影响?若不正交, 那会不会在梯度趋近的时候相互影响影响效率?
各个变量之间的相关性???求最小值的时候能不能利用相关性。







































