##################################################################

#Imports

##################################################################

import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
import pickle

##################################################################

#Data

##################################################################


data_file = pd.read_table(r'ratings.dat', sep = '::', header=None)


##################################################################

#Imports

##################################################################


users = np.unique(data_file[0])
movies = np.unique(data_file[1])
 
number_of_rows = len(users)
number_of_columns = len(movies)

movie_indices, user_indices = {}, {}
 
for i in range(len(movies)):
    movie_indices[movies[i]] = i
    
for i in range(len(users)):
    user_indices[users[i]] = i

#scipy sparse matrix to store the 1M matrix
V = sp.lil_matrix((number_of_rows, number_of_columns))

#adds data into the sparse matrix
for line in data_file.values:
    u, i , r , gona = map(int,line)
    V[user_indices[u], movie_indices[i]] = r

#as these operations consume a lot of time, it's better to save processed data 
with open('movielens_1M.pickle', 'wb') as handle:
    pickle.dump(V, handle)

#as these operations consume a lot of time, it's better to save processed data 
#gets SVD components from 10M matrix
u,s, vt = svds(V, k = 500)
 
with open('movielens_1M_svd_u.pickle', 'wb') as handle:
    pickle.dump(u, handle)
with open('movielens_1M_svd_s.pickle', 'wb') as handle:
    pickle.dump(s, handle)
with open('movielens_1M_svd_vt.pickle', 'wb') as handle:
    pickle.dump(vt, handle)

s_diag_matrix = np.zeros((s.shape[0], s.shape[0]))

for i in range(s.shape[0]):
    s_diag_matrix[i,i] = s[i]

X_lr = np.dot(np.dot(u, s_diag_matrix), vt)

print X_lr