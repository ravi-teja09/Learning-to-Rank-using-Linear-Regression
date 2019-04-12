# Learning-to-Rank-using-Linear-Regression
**University at Buffalo - CSE574: Introduction to Machine Learning**
<p>Project 1.2</p>

### Overview
The goal of this project is to use machine learning to solve a problem that arises in Information Retrieval,
one known as the Learning to Rank (LeToR) problem. We formulate this as a problem of linear regression
where we map an input vector x to a real-valued scalar target y(x;w).
There are two tasks:
1. Train a linear regression model on LeToR dataset using a closed-form solution.
2. Train a linear regression model on the LeToR dataset using stochastic gradient descent (SGD).
The LeToR training data consists of pairs of input values x and target values t. The input values are
real-valued vectors (features derived from a query-document pair). The target values are scalars (relevance
labels) that take one of three values 0, 1, 2: the larger the relevance label, the better is the match between
query and document. Although the training target values are discrete we use linear regression to obtain real
values which is more useful for ranking (avoids collision into only three possible values).
