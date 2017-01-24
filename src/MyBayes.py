from pomegranate import BayesianNetwork
from copy import deepcopy
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import networkx as nx


class MyBayes(object):

    def __init__(self, feature_names, algorithm='chow-liu'):
        """
        :param algorithm:
         type of algorithm to use, possible values:
         * chow-liu - build graph using Chow-Liu algorithm (TAN, according to WEKA)
         * naive - naive Bayes classifier - every node has one root - label node
        """
        self.le = preprocessing.LabelEncoder()
        self.formatted_labels, self.state_names, self.model = None, None, None
        self.state_names = deepcopy(feature_names)
        self.state_names.insert(0, "label")
        self.algorithm = algorithm

    def __str__(self):
        return "MyBayes(algorithm='{}', states={})".format(self.algorithm, len(self.state_names))

    def fit(self, X_train, y_train, sequence_length_train):
        if self.algorithm in ['chow-liu', 'tan']:
            self.fit_chow_liu(X_train, y_train, sequence_length_train)
        elif self.algorithm == 'naive':
            self.fit_naive(X_train, y_train, sequence_length_train)

    def fit_chow_liu(self, X_train, y_train, sequence_length_train):
        # TODO: use sequence_length_train
        self.formatted_labels = self.le.fit_transform(y_train)
        self.formatted_labels = self.formatted_labels.reshape(self.formatted_labels.shape[0], 1)
        X = np.concatenate((self.formatted_labels, X_train), axis=1)
        self.model = BayesianNetwork.from_samples(X, algorithm='chow-liu', state_names=self.state_names, root=0)

    def fit_naive(self, X_train, y_train, sequence_length_train):
        self.formatted_labels = self.le.fit_transform(y_train)
        self.formatted_labels = self.formatted_labels.reshape(self.formatted_labels.shape[0], 1)
        graph = nx.DiGraph()
        for i in range(1, len(self.state_names)):
            graph.add_edge((0,), (i,))
        X = np.concatenate((self.formatted_labels, X_train), axis=1)
        self.model = BayesianNetwork.from_samples(X, algorithm='exact', state_names=self.state_names, root=0,
                                                  constraint_graph=graph)

    @classmethod
    def get_probable_class(clf, probabilites):
        values = list(probabilites.values())
        keys = list(probabilites.keys())
        return int(keys[values.index(max(values))])

    def predict(self, X_test, sequence_length_test):
        if self.algorithm in ['chow-liu', 'tan', 'naive']:
            return self.predict_bayes_net(X_test, sequence_length_test)

    def predict_bayes_net(self, X_test, sequence_length_test):
        # TODO: use sequence_length_test
        results = []

        for sample in X_test:
            mapped_sample = dict(zip(self.state_names[:-1], sample))
            beliefs = self.model.predict_proba(mapped_sample, check_input=False)
            graph = dict(zip([state.name for state in self.model.states], beliefs))
            label_probabilities = graph['label'].parameters[0]
            results.append(self.le.classes_[MyBayes.get_probable_class(label_probabilities)])

        return results

    def plot_graph(self):
        plt.figure(figsize=(16, 8))
        self.model.plot()
        plt.show()
