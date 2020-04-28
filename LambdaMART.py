from math import exp, log
from collections import defaultdict
import datetime
import pickle

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor


FOLDER = './l2r/'
PATH_TRAIN  = FOLDER + 'train.txt'
PATH_TEST   = FOLDER + 'test.txt'
PATH_SAMPLE = FOLDER + 'sample.made.fall.2019'


def count_dcg(docs, k=np.Inf):
    '''
    Count DCG@k for a query
    '''
    docs.sort(key=lambda tup: tup[1], reverse=True)
    
    dcg = 0
    for i, doc in enumerate(docs):
        dcg += (2**doc[1] - 1) / log(2 + i, 2)
        if i >= k:
            break
        
    return dcg


def count_ideal_dcgs(query_docs, k=np.Inf):
    '''
    Count ideal DCG@k
    Return dict: qid -> ideal DCG value
    '''
    ideal_dcgs = dict()
    for query in query_docs.keys():
        ideal_dcgs[query] = count_dcg(query_docs[query], k)
    
    return ideal_dcgs


def count_delta_ndcg(pair, labels, positions, ideal_dcgs):
    '''
    Count delta ndcg value for a document pair
    '''
    i = positions[pair[1]]
    j = positions[pair[2]]
    
    label_i = labels[pair[1]]
    label_j = labels[pair[2]]
    
    dcg      = (2**label_i - 1) / log(1 + i, 2) + (2**label_j - 1) / log(1 + j, 2)
    dcg_swap = (2**label_i - 1) / log(1 + j, 2) + (2**label_j - 1) / log(1 + i, 2)
    
    delta_ndcg = abs((dcg - dcg_swap) / ideal_dcgs[pair[0]])
    
    return delta_ndcg


def group_enumerate(groups):
    '''
    Initiate docs positions for each query
    in appearence order before a start of learning
    '''
    positions = np.zeros(groups.shape, dtype=int)
    
    unique_groups = np.unique(groups)
    for group in unique_groups:
        mask = (groups==group)
        positions[mask] = np.arange(np.sum(mask)) + 1
    
    return positions

    
def group_argsort(scores, groups):
    '''
    Get vector of docs positions by model scores
    for delta_ndcg count
    '''
    positions = np.zeros(scores.shape, dtype=int)
    
    unique_groups = np.unique(groups)
    for group in unique_groups:
        mask = (groups==group)
        positions[mask] = np.sum(mask) - np.argsort(np.argsort(scores[mask]))
    
    return positions


def group_sort(scores, groups):
    '''
    Get vector of docs positions by model scores
    to make submit file
    '''
    ids = np.arange(scores.shape[0])
    
    unique_groups = np.unique(groups)
    for group in unique_groups:
        mask = (groups==group)
        inds = np.argsort(scores[mask])
        ids[mask] = ids[mask][inds[::-1]] + 1
    
    return ids


def count_lambda_(pair, labels, positions, ideal_dcgs, scores):
    '''
    count lambda value for a doc pair
    '''
    s_i = scores[pair[1]]
    s_j = scores[pair[2]]
    
    delta_ndgc = count_delta_ndcg(pair, labels, positions, ideal_dcgs)
    lambda_ij = -delta_ndgc / (1 + exp( s_i - s_j ))
    
    return lambda_ij


def count_mean_ndcg_k_for_dataset(qids, labels, scores, ideal_dcgs):
    '''
    Function to track target metric nDCG through learning process
    '''
    ndcg = 0
    n = 0
    
    positions = group_argsort(scores, qids)
    dcgs = (np.power(2, labels) - 1) / np.log2(1 + positions)
    
    unique_qids = np.unique(qids)
    for qid in unique_qids:
        mask = (qids==qid)
        if np.sum(mask)==1:
            continue
        if ideal_dcgs[qid]==0:
            continue
        ndcg += np.sum(dcgs[(mask)&(positions<=5)]) / ideal_dcgs[qid]
        n += 1
    
    return ndcg / n


def find_const_features(a):
    '''
    Get list of ids of const columns of 2d array
    '''
    const_features = []
    for i, col in enumerate(a.T):
        if np.var(col)==0:
            const_features += [i]
    
    return const_features


class LambdaMART:
    def __init__(self, n_trees=100, learning_rate=0.1):
        self.n_trees = n_trees
        self.learning_rate = learning_rate
        self.trees = []
    
    def fit(self, path='./train.txt/'):
        # determine needed attributes
        self.get_data_shape_attributes_(path)
        self.get_data_features_(path)
        self.get_pairs_()
        
        # drop const features
        self.const_features = find_const_features(self.X)
        self.X = np.delete(self.X, self.const_features, axis=1)
        
        # count ideal dcg for every query
        self.ideal_dcgs   = count_ideal_dcgs(self.query_docs)
        self.ideal_dcgs_k = count_ideal_dcgs(self.query_docs, 5)
        
        lambdas = np.zeros(self.n_rows)
        positions = group_enumerate(self.qids)
        for k in range(self.n_trees):
            # count lambdas
            for pair in self.pairs:
                lambda_ij = count_lambda_(pair, self.labels, positions, self.ideal_dcgs, self.scores)
                lambdas[pair[1]] -= lambda_ij
                lambdas[pair[2]] += lambda_ij   
            
            # build a regression tree
            tree = DecisionTreeRegressor(max_depth=10)
            tree.fit(self.X, lambdas)

            # update model state
            self.trees.append(tree)
            prediction = tree.predict(self.X)
            self.scores += prediction * self.learning_rate

            lambdas = np.zeros(self.n_rows)
            positions = group_argsort(self.scores, self.qids)

            # print learning info
            if ((k+1)%50)==0:
                print('%s\t%s-th tree is building...' % (datetime.datetime.now(), k))
                ndcg_cur = count_mean_ndcg_k_for_dataset(self.qids, self.labels, self.scores, self.ideal_dcgs_k)
                print('current train ndcg:', ndcg_cur)

                # save model
                with open(f'lambdaMART_{k+1}', 'wb') as f:
                    pickle.dump(self.trees, f)


    def predict(self, path='./test.txt/'):
        # determine needed attributes
        self.get_data_shape_attributes_(path)
        self.get_data_features_(path)

        # drop const features
        self.X = np.delete(self.X, self.const_features, axis=1)

        # for k in range(self.n_trees):
        for k in range(len(self.trees)):
            tree = self.trees[k]
            prediction = tree.predict(self.X)
            self.scores += prediction * self.learning_rate

        return self.scores


    def get_data_shape_attributes_(self, path):
        '''
        Get number of rows and features in dataset
        ''' 
        self.max_feature_ind = 0
        with open(path, 'r') as f:
            for i, row in enumerate(f):
                max_row_ind = max([int(x.split(':')[0]) for x in row.split()[2:]])
                if max_row_ind > self.max_feature_ind:
                    self.max_feature_ind = max_row_ind

        self.n_rows = i + 1


    def get_data_features_(self, path):
        '''
        Read data
        '''
        self.qids   = np.zeros(self.n_rows, dtype=int)
        self.labels = np.zeros(self.n_rows, dtype=float) # assessors' labels
        self.scores = np.zeros(self.n_rows, dtype=float) # model scores
        
        self.query_docs = defaultdict(list)
        self.X = -np.ones((self.n_rows, self.max_feature_ind))

        with open(path, 'r') as f:
            for doc_id, row in enumerate(f):
                grade, query_id, features = row.split(' ', 2)

                grade = int(grade)
                query_id = int(query_id.split(':')[1])

                # update vectors
                self.qids[doc_id] = query_id
                self.labels[doc_id] = grade

                # update query_docs
                self.query_docs[query_id].append((doc_id, grade))
    
                # update feature matrix
                self.X[doc_id, [int(x.split(':')[0])-1 for x in features.split()]] = [float(x.split(':')[1]) for x in features.split()]


    def get_pairs_(self):
        '''
        Get documents pairs
        Result: list of tuples: (query_id, index of better doc, index of worse doc)
        '''
        self.pairs = []
        for query_id, docs in self.query_docs.items():
            if len(docs)==1:
                continue
            else:
                docs.sort(key=lambda tup: tup[1], reverse=True)
                for i in range(len(docs)):
                    for j in range(i+1, len(docs)):
                        if docs[i][1] > docs[j][1]:
                            self.pairs.append((query_id, docs[i][0], docs[j][0]))


def main():
    # learn model
    model = LambdaMART(n_trees=2000, learning_rate=1.)
    model.fit(PATH_TRAIN)
    model.predict(PATH_TEST)
    
    # make submit
    sample = pd.read_csv(PATH_SAMPLE)
    sample['DocumentId'] = group_sort(model.scores, model.qids)
    sample.to_csv('submit.txt', index=False)


if __name__ == "__main__":
    main()
