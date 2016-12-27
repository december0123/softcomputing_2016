from src.datasets import LearningSetFactory
from seqlearn.hmm import MultinomialHMM
from sklearn.metrics import accuracy_score

from src.MyBayes import MyBayes

def train_and_test_markov(X_train, y_train, sequence_length_train, X_test, y_test,
                          sequence_length_test, *args, **kwargs):
    clf = MultinomialHMM()
    print("X_train")
    print(len(X_train))
    print("y_train")
    print(len(y_train))
    print("sequence_length_train")
    print(len(sequence_length_train))
    clf.fit(X_train, y_train, sequence_length_train)
    print("Training {}".format(clf))
    print(len(y_test))
    y_pred = clf.predict(X_test, sequence_length_test)
    print("Accuracy: {0:.2f}".format(100 * accuracy_score(y_pred, y_test)))


def train_and_test_bayes(X_train, y_train, sequence_length_train, X_test, y_test,
                          sequence_length_test, feature_names, *args, **kwargs):
    print("Bayes network training")
    clf = MyBayes(feature_names)
    clf.fit(X_train, y_train, sequence_length_train)
    print("Training {}".format(clf))
    print(len(y_test))
    y_pred = clf.predict(X_test, sequence_length_test)
    print("Accuracy: {0:.2f}".format(100 * accuracy_score(y_pred, y_test)))


def main():
    learning_factory = LearningSetFactory()
    # train_and_test_bayes(*learning_factory.get_train_test_data(LearningSetFactory.DataSource.activity_recognition))
    # train_and_test_bayes(*learning_factory.get_train_test_data(LearningSetFactory.DataSource.breast_cancer))
    train_and_test_markov(*learning_factory.get_train_test_data(LearningSetFactory.DataSource.breast_cancer))
    train_and_test_markov(*learning_factory.get_train_test_data(LearningSetFactory.DataSource.sequenced_breast_cancer))

if __name__ == "__main__":
    main()