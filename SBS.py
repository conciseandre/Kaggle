
# coding: utf-8

# In[2]:

from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


class SBS():
    
    def __init__(self, estimator, k_features,
                 scoring = accuracy_score,
                 test_size=0.25, random_state =1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state
    
    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                           random_state = self.random_state)
        
        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)

        self.scores_ = [score]
        
        while dim > self.k_features:
            scores = []
            subsets = []
            
            for p in combinations(self.indices_, r = dim-1):
                score = self._calc_score(X_train, y_train, X_test, y_test, p)
                
                scores.append(score)
                subsets.append(p)
                
            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim = dim - 1
                
            self.scores_.append(scores[best])
        self.k_score = self.scores_[-1]
            
        return self
        
    def transform(self, X):
        return X[:, self.indices]
        
    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score
    def plot_features(self):
        self.k_feat = [len(k) for k in self.subsets_]
        plt.plot(self.k_feat, self.scores_, marker = 'o')
        plt.ylim(0.7, 1.1)
        plt.ylabel('accuracy')
        plt.xlabel('num features')
        plt.grid()
        plt.show()
    def param_space(self,df):
        import random
        best = np.where(self.scores_ == max(self.scores_))
        best[0].tolist()
        best_choice = random.choice(best[0])
        self.k_i = list(self.subsets_[int(best_choice)])
        print("Recommended Param space is :", df.columns[self.k_i]) #requires class label in column 0


        


# In[ ]:



