from src.datasets import LearningSetFactory
from seqlearn.hmm import MultinomialHMM
from sklearn.metrics import accuracy_score
from pomegranate import BayesianNetwork
from sklearn.feature_extraction import DictVectorizer
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

def train_and_test_markov(X_train, y_train, sequence_length_train, X_test, y_test,
                          sequence_length_test, *args, **kwargs):
    clf = MultinomialHMM()
    clf.fit(X_train, y_train, sequence_length_train)
    print("Training {}".format(clf))
    print(len(y_test))
    y_pred = clf.predict(X_test, sequence_length_test)
    print("Accuracy: {0:.2f}".format(100 * accuracy_score(y_pred, y_test)))


def train_and_test_bayes(X_train, y_train, sequence_length_train, X_test, y_test,
                          sequence_length_test, feature_names, *args, **kwargs):
    print("Bayes network training")
    le = preprocessing.LabelEncoder()
    formatted_labels = le.fit_transform(y_train)
    formatted_labels = formatted_labels.reshape(formatted_labels.shape[0], 1)
    print("Classes", le.classes_)
    X = np.concatenate((X_train, formatted_labels), axis=1)
    print("After merge ", X.shape)
    state_names = deepcopy(feature_names)
    state_names.extend("Cancer_type")
    model = BayesianNetwork.from_samples(X, algorithm='chow-liu', state_names=state_names)

    plt.figure(figsize=(16, 8))
    model.plot()
    plt.show()

    formatted_test_labels = le.fit_transform(y_test)
    formatted_test_labels = formatted_test_labels.reshape(formatted_labels.shape[0], 1)
    Test = np.concatenate((X_train, formatted_labels), axis=1)


    print(model.predict_proba({ state_names[0]: 0}))
    raise NotImplementedError


def main():
    learning_factory = LearningSetFactory()
    train_and_test_bayes(*learning_factory.get_train_test_data(LearningSetFactory.DataSource.breast_cancer))
    train_and_test_markov(*learning_factory.get_train_test_data(LearningSetFactory.DataSource.breast_cancer))
    train_and_test_markov(*learning_factory.get_train_test_data(LearningSetFactory.DataSource.activity_reccognition))

if __name__ == "__main__":
    main()