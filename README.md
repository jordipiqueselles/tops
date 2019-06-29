# ToPs (Tree Of Predictors)

This is a public implementation of the ToPs predictive algorithm.
This implementation is based on the paper 
_"Tops: Ensemble learning with trees of predictors"_, writed by
Jinsung Yoon, William R Zame, and Mihaela van der Schaar.


## How does ToPs work?

ToPs is created by constructing a tree, but with a predictor associated 
to each node. The tree implicitly segments the feature space 
by splitting the dataset recursively and this creates subsets from
different regions of this space. Then, the predictors can 
specialize in learning only the characteristics of
these subsets of instances. The overall prediction of the tree 
for an instance is obtained by aggregating the results of 
the predictors found along the unique path from the root to a 
leaf for this instance.