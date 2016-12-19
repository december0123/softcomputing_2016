from pomegranate import BayesianNetwork
from copy import deepcopy
import numpy as np
from sklearn import preprocessing


class MyBayes(object):

    def __init__(self, feature_names):
        self.le = preprocessing.LabelEncoder()
        self.formatted_labels, self.state_names, self.model = None, None, None
        self.state_names = deepcopy(feature_names)
        self.state_names.append("label")

    def fit(self, X_train, y_train, sequence_length_train):
        self.formatted_labels = self.le.fit_transform(y_train)
        self.formatted_labels = self.formatted_labels.reshape(self.formatted_labels.shape[0], 1)
        X = np.concatenate((X_train, self.formatted_labels), axis=1)
        self.model = BayesianNetwork.from_samples(X, algorithm='chow-liu', state_names=self.state_names)

    def predict(self, X_test, sequence_length_test):
        def get_probable_class(probabilites):
            values = list(probabilites.values())
            keys = list(probabilites.keys())
            return int(keys[values.index(max(values))])

        results = []

        for sample in X_test:
            mapped_sample = dict(zip(self.state_names[:-1], sample))
            print(mapped_sample)
            beliefs = self.model.predict_proba(mapped_sample)
            graph = dict(zip([state.name for state in self.model.states], beliefs))
            label_probabilities = graph['label'].parameters[0]
            results.append(self.le.classes_[get_probable_class(label_probabilities)])
            print(results)

        return results
