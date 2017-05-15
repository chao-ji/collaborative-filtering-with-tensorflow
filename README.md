# collaborative-filtering-with-tensorflow

This work was inspired by [TF-recomm](https://github.com/songgc/TF-recomm). Rather than forking this repo, I started over by refactoring the code to try to conform to the convention used in scikit-learn's `Estimator` (e.g., `fit` and `predict` method, the underscore at the end of learned attributes). 

## Collaborative Filering based on Latent Factor Model

This implementation minimizes the loss function defined in Eq. 5 in this article: [Matric Factorization Techniques For Recommender Systems](https://datajobs.com/data-science-repo/Recommender-Systems-%5BNetflix%5D.pdf)


## MovieLens Dataset

The model was evaluated using the [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) dataset. In addition to computing RMSE as the evaluation metric, I performed on exploratory data analysis on the movie and users embeddings. Some quick results:

1. Group pairs of movies by the number of shared genre types (e.g. movie 1: [War|Action], movie 2: [Action|Thiller], shared genre types is  1), and compute the avarege pairwise cosine similarity of movie embeddings for each group.
<img src="https://github.com/chao-ji/collaborative-filtering-with-tensorflow/blob/master/shared%20genre%20types.png" width="600">

2. Scatter plot of user embeddings in 2D space (red: Male users, blue: female users)
<img src="https://github.com/chao-ji/collaborative-filtering-with-tensorflow/blob/master/MandF.png" width="600">

3. Users are categorized into **age groups**. Compute inter-group distances (i.e. average of cosine distance between user *i* and user *j* from different groups)
<img src="https://github.com/chao-ji/collaborative-filtering-with-tensorflow/blob/master/age%20group.png" width="600">

4. Users are categorized into **occupation groups**. Compute inter-group distances (i.e. average of cosine distance between user *i* and user *j* from different groups)
<img src="https://github.com/chao-ji/collaborative-filtering-with-tensorflow/blob/master/occupation%20group.png">


## What's next:

* Alternating Least Square implementaion: ALS algorithm proves to be more robust. It works by first fixing the user embeddings first and estimating the optimal item embeddings; and then fixing the item embeddings and estimate user embeddings

* MovieLens 1M dataset contains ratings data ranging from 1995 2003. The 20M dataset is much larger and has a much longer temporal span (up to 2016). This is very useful for investigating incorporating the temporal effects in the collaborative filtering model.
