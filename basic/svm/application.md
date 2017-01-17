###Application

1. Q: What is the difference between a one-vs-all and a one-vs-one SVM classifier?
Does the one-vs-all mean one classifier to classify all types / categories of the new image and one-vs-one mean each type / category of new image classify with different classifier (each category is handled by special classifier)?

A: 
10
down vote
The difference is the number of classifiers you have to learn, which strongly correlates with the decision boundary they create.

Assume you have N different classes. **One vs all** will train one classifier per class in total N classifiers. For class i it will assume i-labels as positive and the rest as negative. This often leads to imbalanced datasets meaning generic SVM might not work, but still there are some workarounds.

In **one vs one** you have to train a separate classifier for each different pair of labels. This leads to N(N−1)2 classifiers. This is much less sensitive to the problems of imbalanced datasets but is much more computationally expensive.












