from pomegranate import BayesianNetwork, DiscreteDistribution, State
from copy import deepcopy
import numpy as np
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt


class MyBayes(object):

    def __init__(self, feature_names, algorithm='chow-liu'):
        """
        :param algorithm:
         type of algorithm to use, possible values:
         * chow-liu - build graph using Chow-Liu algorithm
         * tan - tree augmented Naive Bayes (more information:
            http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.55.7726&rep=rep1&type=pdf)
            it creates graph using Chow-Liu algorithm, then appends class edge as parent for all remaining edges
         * naive - Naive Bayes, used classifier: MultinomialNB from sklearn
        """
        self.le = preprocessing.LabelEncoder()
        self.formatted_labels, self.state_names, self.model = None, None, None
        self.state_names = deepcopy(feature_names)
        self.state_names.append("label")
        self.algorithm = algorithm

    def __str__(self):
        return "MyBayes(algorithm='{}', states={})".format(self.algorithm, len(self.state_names))

    def fit(self, X_train, y_train, sequence_length_train):
        if self.algorithm == 'chow-liu':
            self.fit_chow_liu(X_train, y_train, sequence_length_train)
        if self.algorithm == 'tan':
            self.fit_tan(X_train, y_train, sequence_length_train)
        elif self.algorithm == 'naive':
            self.fit_naive(X_train, y_train, sequence_length_train)

    def fit_chow_liu(self, X_train, y_train, sequence_length_train):
        # TODO: use sequence_length_train
        self.formatted_labels = self.le.fit_transform(y_train)
        self.formatted_labels = self.formatted_labels.reshape(self.formatted_labels.shape[0], 1)
        X = np.concatenate((X_train, self.formatted_labels), axis=1)
        self.model = BayesianNetwork.from_samples(X, algorithm='chow-liu', state_names=self.state_names)

    def fit_tan(self, X_train, y_train, sequence_length_train):
        self.formatted_labels = self.le.fit_transform(y_train)
        self.formatted_labels = self.formatted_labels.reshape(self.formatted_labels.shape[0], 1)
        X = np.concatenate((X_train, self.formatted_labels), axis=1)
        self.model = BayesianNetwork.from_samples(X, algorithm='chow-liu', state_names=self.state_names,
                                                  root=len(self.state_names)-1)
        for i in range(len(self.model.states[:-1])):
            adding_needed = True
            for edge in self.model.edges:
                if 'label' in [edge[0].name, edge[1].name]:
                    adding_needed = False
                    break
            if adding_needed:
                self.model.add_edge(self.model.states[-1], self.model.states[i])
        self.model.bake()

    def fit_naive(self, X_train, y_train, sequence_length_train):
        self.model = MultinomialNB()
        self.model.fit(X_train, y_train)

    @classmethod
    def get_probable_class(clf, probabilites):
        values = list(probabilites.values())
        keys = list(probabilites.keys())
        return int(keys[values.index(max(values))])

    def predict(self, X_test, sequence_length_test):
        if self.algorithm in ['chow-liu', 'tan']:
            return self.predict_chow_liu(X_test, sequence_length_test)
        elif self.algorithm == 'naive':
            return self.predict_naive(X_test, sequence_length_test)

    def predict_chow_liu(self, X_test, sequence_length_test):
        # TODO: use sequence_length_test
        results = []

        for sample in X_test:
            mapped_sample = dict(zip(self.state_names[:-1], sample))
            beliefs = self.model.predict_proba(mapped_sample, check_input=False)
            graph = dict(zip([state.name for state in self.model.states], beliefs))
            label_probabilities = graph['label'].parameters[0]
            results.append(self.le.classes_[MyBayes.get_probable_class(label_probabilities)])

        return results

    def predict_naive(self, X_test, sequence_length_test):
        return list(self.model.predict(X_test))

    def plot_graph(self):
        plt.figure(figsize=(16, 8))
        self.model.plot()
        plt.show()
