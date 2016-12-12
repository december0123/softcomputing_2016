from src.datasets import LearningSetFactory
from seqlearn.hmm import MultinomialHMM
from sklearn.metrics import accuracy_score


def train_and_test_markov(X_train, y_train, sequence_length_train, X_test, y_test, sequence_length_test):
    clf = MultinomialHMM()
    clf.fit(X_train, y_train, sequence_length_train)
    print("Training {}".format(clf))
    print(len(y_test))
    y_pred = clf.predict(X_test, sequence_length_test)
    print("Accuracy: {0:.2f}".format(100 * accuracy_score(y_pred, y_test)))


def train_and_test_bayes():
    raise NotImplementedError


def main():
    learning_factory = LearningSetFactory()
    train_and_test_markov(*learning_factory.get_train_test_data(LearningSetFactory.DataSource.breast_cancer))
    train_and_test_markov(*learning_factory.get_train_test_data(LearningSetFactory.DataSource.activity_reccognition))

if __name__ == "__main__":
    main()