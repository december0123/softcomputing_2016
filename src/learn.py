from src.datasets import LearningSetFactory
from seqlearn.hmm import MultinomialHMM
from sklearn.metrics import accuracy_score
from pomegranate import BayesianNetwork
from sklearn.feature_extraction import DictVectorizer
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np

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
    vectorizer = DictVectorizer(sparse=False)
    labels = [{label: 1.0} for label in y_train]
    formatted_labels = vectorizer.fit_transform(labels)
    print("Classes", vectorizer.get_feature_names())
    print("X train shape", X_train.shape)
    print("Formatted labels shape", formatted_labels.shape)
    X = np.concatenate((X_train, formatted_labels), axis=1)
    print("After merge ", X.shape)
    state_names = deepcopy(feature_names)
    state_names.extend(vectorizer.get_feature_names())
    model = BayesianNetwork.from_samples(X, algorithm='chow-liu', state_names=state_names)

    plt.figure(figsize=(16, 8))
    model.plot()
    plt.show()
    print(model.predict_proba({}))
    raise NotImplementedError


def main():
    learning_factory = LearningSetFactory()
    train_and_test_bayes(*learning_factory.get_train_test_data(LearningSetFactory.DataSource.breast_cancer))
    train_and_test_markov(*learning_factory.get_train_test_data(LearningSetFactory.DataSource.breast_cancer))
    train_and_test_markov(*learning_factory.get_train_test_data(LearningSetFactory.DataSource.activity_reccognition))

if __name__ == "__main__":
    main()