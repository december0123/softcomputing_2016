from src.datasets import LearningSetFactory
from seqlearn.hmm import MultinomialHMM
from sklearn.metrics import accuracy_score
#import weka.core.jvm as jvm
from src.MyBayes import MyBayes
#from src.WekaBayes import WekaBayes
import matplotlib.pyplot as plt
import numpy as np
import time


def markov_experiments(learning_factory):
    alpha = 0.01
    decodes = ['viterbi', 'bestfirst']
    result = {}
    for decode in decodes:
        #print("Decode: {}".format(decode))
        nonsequenced = []
        sequenced = []

        for i in range(100):
            nonseq_result = train_and_test_markov(decode, alpha, *learning_factory.get_train_test_data(LearningSetFactory.DataSource.breast_cancer))
            nonsequenced.append(nonseq_result)

            seq_result = train_and_test_markov(decode, alpha, *learning_factory.get_train_test_data(LearningSetFactory.DataSource.sequenced_breast_cancer))
            sequenced.append(seq_result)

        result[decode] = [nonsequenced, sequenced]
    visualize_markov(result)


def visualize_markov(result):
    nv = parse_markov_data(result['viterbi'][0]) # nonsequenced viterbi
    sv = parse_markov_data(result['viterbi'][1]) # sequenced viterbi
    nb = parse_markov_data(result['bestfirst'][0]) # nonsequenced bestfirst
    sb = parse_markov_data(result['bestfirst'][1]) # sequenced bestfirst

    mins, maxes, means, std = generate_markov_statistics(nv, sv, nb, sb, "fit")
    plot_markov(mins, maxes, means, std)
    mins, maxes, means, std = generate_markov_statistics(nv, sv, nb, sb, "pred")
    plot_markov(mins, maxes, means, std)
    mins, maxes, means, std = generate_markov_statistics(nv, sv, nb, sb, "accu")
    plot_markov(mins, maxes, means, std)


def plot_markov(mins, maxes, means, std):
    print("Mins: {}".format(mins))
    print("Maxes: {}".format(maxes))
    print("Means: {}".format(means))
    print("Std: {}".format(std))
    plt.errorbar(np.arange(4), means, std, fmt='ok', lw=3)
    plt.errorbar(np.arange(4), means, [means - mins, maxes - means], fmt='.k', ecolor="gray", lw=1)
    plt.xlim(-1, 4)
    plt.xticks(np.arange(4), ["nonseq viterbi", "seq viterbi", "nonseq bestfirst", "seq bestfirst"], rotation=15)
    plt.show()


def generate_markov_statistics(nv, sv, nb, sb, data_type):
    mins = np.array([np.min(nv[data_type]), np.min(sv[data_type]), np.min(nb[data_type]), np.min(sb[data_type])])
    maxes = np.array([np.max(nv[data_type]), np.max(sv[data_type]), np.max(nb[data_type]), np.max(sb[data_type])])
    means = np.array([np.mean(nv[data_type]), np.mean(sv[data_type]), np.mean(nb[data_type]), np.mean(sb[data_type])])
    std = np.array([np.std(nv[data_type]), np.std(sv[data_type]), np.std(nb[data_type]), np.std(sb[data_type])])

    return mins, maxes, means, std


def parse_markov_data(data):
    fit_time = np.array([i[0] for i in data])
    pred_time = np.array([i[1] for i in data])
    accuracy = np.array([i[2] for i in data])

    return {"fit": fit_time, "pred": pred_time, "accu": accuracy}


def train_and_test_markov(decode, alpha, X_train, y_train, sequence_length_train, X_test, y_test,
                          sequence_length_test, *args, **kwargs):
    clf = MultinomialHMM(decode=decode, alpha=alpha)
    #print("Training {}".format(clf))
    start = time.time()
    clf.fit(X_train, y_train, sequence_length_train)
    mid = time.time()
    y_pred = clf.predict(X_test, sequence_length_test)
    stop = time.time()
    accuracy = 100 * accuracy_score(y_pred, y_test)
    fit_time = 1000 * (mid - start)
    pred_time = 1000 * (stop - mid)
    #print("Fit time: {:.3f}ms, Predict time: {:.3f}ms, Accuracy: {:.2f}".format(fit_time, pred_time, accuracy))
    return (fit_time, pred_time, accuracy)


def train_and_test_bayes(algorithm, X_train, y_train, sequence_length_train, X_test, y_test,
                         sequence_length_test, feature_names, *args, **kwargs):
    print("Bayes network training")
    clf = MyBayes(feature_names, algorithm=algorithm)
    clf.fit(X_train, y_train, sequence_length_train)
    print("Training {}".format(clf))
    y_pred = clf.predict(X_test, sequence_length_test)
    print("Accuracy: {0:.2f}".format(100 * accuracy_score(y_pred, y_test)))
    #print(zip(y_test,y_pred))
    #print(y_pred)
    r2r = r2n = n2n = n2r = 0
    for x,y in zip(y_test, y_pred):
        print(x, '->', y)
        if x=='recurrence-events' and y=='recurrence-events':
            r2r += 1
        if x=='recurrence-events' and y=='no-recurrence-events':
            r2n += 1
        if x=='no-recurrence-events' and y=='no-recurrence-events':
            n2n += 1
        if x=='no-recurrence-events' and y=='recurrence-events':
            n2r += 1
    print('r2r:', r2r)
    print('r2n:', r2n)
    print('n2n:', n2n)
    print('n2r:', n2r)
    # print(r2r, r2n, n2n, n2r)
    clf.plot_graph()


def train_and_test_weka_bayes(algorithm, train_set, test_set, labels, *args, **kwargs):
    print("Bayes network training")
    clf = WekaBayes(labels, algorithm=algorithm)
    clf.fit(train_set)
    print("Training {}".format(clf))
    y_pred = clf.predict(test_set)
    y_test = [sample.get_value(sample.class_index) for _, sample in enumerate(test_set)]
    print("Accuracy: {0:.2f}".format(100 * accuracy_score(y_pred, y_test)))


def main():
    learning_factory = LearningSetFactory()
    #train_and_test_bayes(*learning_factory.get_train_test_data(LearningSetFactory.DataSource.activity_recognition))
    train_and_test_bayes('naive', *learning_factory.get_train_test_data(LearningSetFactory.DataSource.breast_cancer))
    train_and_test_bayes('chow-liu', *learning_factory.get_train_test_data(LearningSetFactory.DataSource.breast_cancer))
    '''
    jvm.start()

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
    jvm.stop()
    '''
    #train_and_test_markov(*learning_factory.get_train_test_data(LearningSetFactory.DataSource.activity_recognition))
    #train_and_test_markov(*learning_factory.get_train_test_data(LearningSetFactory.DataSource.sequenced_activity_recognition))

    #markov_experiments(learning_factory)

if __name__ == "__main__":
    main()
