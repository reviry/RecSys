import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

def pivot(ratings):
    ratings = pd.pivot_table(ratings, values='rating', index=['user_id'], columns=['movie_id'])
    ratings = ratings.fillna(0)
    ratings = ratings.as_matrix()
    return ratings

def extract(ratings):
    sub_ratings = ratings.loc[random.sample(ratings.index, 2)]
    ratings = ratings.drop(sub_ratings.index)
    return ratings, sub_ratings

def create_matrix(df, ratings):
    for i in ratings.index:
        df.ix[i]['rating'] = ratings.ix[i]['rating']
    ratings = df.copy()
    df['rating'] = 0
    return df, ratings

ratings_headers = ['user_id', 'movie_id', 'rating', 'timestamp']
df = pd.read_table('sample.dat', sep='::', header=None, names=ratings_headers)
ratings = df.copy()
ratings_org = ratings.copy()
ratings_org = pivot(ratings_org)
df['rating'] = 0

print '****************df****************'
print df

print '*************ratings**************'
print ratings

ratings, ratings1 = extract(ratings)
ratings, ratings2 = extract(ratings)
ratings, ratings3 = extract(ratings)
ratings, ratings4 = extract(ratings)
ratings, ratings5 = extract(ratings)

print '*************ratings1**************'
print ratings1
print '*************ratings2**************'
print ratings2
print '*************ratings3**************'
print ratings3
print '*************ratings4**************'
print ratings4
print '*************ratings5**************'
print ratings5

df, ratings1 = create_matrix(df, ratings1)
df, ratings2 = create_matrix(df, ratings2)
df, ratings3 = create_matrix(df, ratings3)
df, ratings4 = create_matrix(df, ratings4)
df, ratings5 = create_matrix(df, ratings5)

ratings1 = pivot(ratings1)
ratings2 = pivot(ratings2)
ratings3 = pivot(ratings3)
ratings4 = pivot(ratings4)
ratings5 = pivot(ratings5)

print '*************ratings**************'
print ratings_org
print '*************ratings1**************'
print ratings1
print '*************ratings2**************'
print ratings2
print '*************ratings3**************'
print ratings3
print '*************ratings4**************'
print ratings4
print '*************ratings5**************'
print ratings5

print '*************ratings**************'
print ratings1 + ratings2 +ratings3 +ratings4 +ratings5
