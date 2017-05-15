# collaborative-filtering-with-tensorflow

This work was inspired by [TF-recomm](https://github.com/songgc/TF-recomm). Rather than forking this repo, I started over by refactoring the code to try to conform to the convention used in scikit-learn's `Estimator` (e.g., `fit` and `predict` method, the underscore at the end of learned attributes). 

## Collaborative Filering based on Latent Factor Model

This implementation minimizes the loss function defined in Eq. 5 in this article: [MATRIX FACTORIZATION TECHNIQUES FOR RECOMMENDER SYSTEMS]https://datajobs.com/data-science-repo/Recommender-Systems-%5BNetflix%5D.pdf


## MovieLens Dataset

The model was evaluated using the [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) dataset. In addition to computing RMSE as the evaluation metric, I performed on exploratory data analysis on the movie and users embeddings. Some quick results:

1. Group pairs of movies by the number of shared genre types (e.g. movie 1: [War|Action], movie 2: [Action|Thiller], shared genre types is  1), and compute the avarege pairwise cosine similarity of movie embeddings for each group.
<img src="https://github.com/chao-ji/collaborative-filtering-with-tensorflow/blob/master/shared%20genre%20types.png">


## What's next:
