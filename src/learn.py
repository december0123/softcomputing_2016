from src.datasets import LearningSetFactory
from seqlearn.hmm import MultinomialHMM
from sklearn.metrics import accuracy_score
import weka.core.jvm as jvm
from src.MyBayes import MyBayes
from src.WekaBayes import WekaBayes


def train_and_test_markov(X_train, y_train, sequence_length_train, X_test, y_test,
                          sequence_length_test, *args, **kwargs):
    clf = MultinomialHMM()
    clf.fit(X_train, y_train, sequence_length_train)
    print("Training {}".format(clf))
    y_pred = clf.predict(X_test, sequence_length_test)
    print("Accuracy: {0:.2f}".format(100 * accuracy_score(y_pred, y_test)))


def train_and_test_bayes(algorithm, X_train, y_train, sequence_length_train, X_test, y_test,
                         sequence_length_test, feature_names, *args, **kwargs):
    print("Bayes network training")
    clf = MyBayes(feature_names, algorithm=algorithm)
    clf.fit(X_train, y_train, sequence_length_train)
    print("Training {}".format(clf))
    y_pred = clf.predict(X_test, sequence_length_test)
    print("Accuracy: {0:.2f}".format(100 * accuracy_score(y_pred, y_test)))


def train_and_test_weka_bayes(algorithm, train_set, test_set, labels, *args, **kwargs):
    print("Bayes network training")
    clf = WekaBayes(labels, algorithm=algorithm)
    clf.fit(train_set)
    print("Training {}".format(clf))
    y_pred = clf.predict(test_set)
    y_test = [sample.get_value(sample.class_index) for _, sample in enumerate(test_set)]
    print("Accuracy: {0:.2f}".format(100 * accuracy_score(y_pred, y_test)))


def main():
    jvm.start()
    learning_factory = LearningSetFactory()
    # train_and_test_bayes(*learning_factory.get_train_test_data(LearningSetFactory.DataSource.activity_recognition))
    train_and_test_bayes('naive', *learning_factory.get_train_test_data(LearningSetFactory.DataSource.breast_cancer))
    train_and_test_bayes('chow-liu', *learning_factory.get_train_test_data(LearningSetFactory.DataSource.breast_cancer))

    train_and_test_weka_bayes('GeneticSearch',
                              *learning_factory.get_train_test_data(LearningSetFactory.DataSource.weka_breast_cancer))
    train_and_test_weka_bayes('HillClimber',
                              *learning_factory.get_train_test_data(LearningSetFactory.DataSource.weka_breast_cancer))
    train_and_test_weka_bayes('K2',
                              *learning_factory.get_train_test_data(LearningSetFactory.DataSource.weka_breast_cancer))
    train_and_test_weka_bayes('LAGDHillClimber',
                              *learning_factory.get_train_test_data(LearningSetFactory.DataSource.weka_breast_cancer))
    train_and_test_weka_bayes('LocalScoreSearchAlgorithm',
                              *learning_factory.get_train_test_data(LearningSetFactory.DataSource.weka_breast_cancer))
    train_and_test_weka_bayes('RepeatedHillClimber',
                              *learning_factory.get_train_test_data(LearningSetFactory.DataSource.weka_breast_cancer))
    train_and_test_weka_bayes('SimulatedAnnealing',
                              *learning_factory.get_train_test_data(LearningSetFactory.DataSource.weka_breast_cancer))
    train_and_test_weka_bayes('TabuSearch',
                              *learning_factory.get_train_test_data(LearningSetFactory.DataSource.weka_breast_cancer))
    train_and_test_weka_bayes('TAN',
                              *learning_factory.get_train_test_data(LearningSetFactory.DataSource.weka_breast_cancer))


    # train_and_test_markov(*learning_factory.get_train_test_data(LearningSetFactory.DataSource.breast_cancer))
    # train_and_test_markov(*learning_factory.get_train_test_data(LearningSetFactory.DataSource.activity_recognition))
    # train_and_test_markov(*learning_factory.get_train_test_data(LearningSetFactory.DataSource.sequenced_breast_cancer))
    # this will take a LONG time
    # train_and_test_markov(*learning_factory.get_train_test_data(LearningSetFactory.DataSource.sequenced_activity_recognition))
    jvm.stop()

if __name__ == "__main__":
    main()
