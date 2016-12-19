from src.datasets import LearningSetFactory
from seqlearn.hmm import MultinomialHMM
from sklearn.metrics import accuracy_score
from pomegranate import BayesianNetwork
import matplotlib.pyplot as plt


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
    model = BayesianNetwork.from_samples(X_train, algorithm='chow-liu', state_names=feature_names)
    plt.figure(figsize=(14, 10))
    model.plot()
    plt.show()

    print(feature_names)
    raise NotImplementedError


def main():
    learning_factory = LearningSetFactory()
    train_and_test_bayes(*learning_factory.get_train_test_data(LearningSetFactory.DataSource.breast_cancer))
    train_and_test_markov(*learning_factory.get_train_test_data(LearningSetFactory.DataSource.breast_cancer))
    train_and_test_markov(*learning_factory.get_train_test_data(LearningSetFactory.DataSource.activity_reccognition))

if __name__ == "__main__":
    main()