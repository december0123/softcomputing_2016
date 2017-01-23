from copy import deepcopy
from weka.classifiers import Classifier


class WekaBayes(object):
    """
    I don't care anymore - Phil Collins

    We don't need no water let the motherfucker burn.
    """

    def __init__(self, labels, algorithm='K2'):
        """
        Waltz
        Tango
        Foxtrot
        """
        self.algorithm = algorithm
        self.labels = deepcopy(labels)
        self.model = None

    def __str__(self):
        return "WekaBayes(algorithm='{}')".format(self.algorithm)

    def fit(self, train_set):
        """
        Foxtrot
        Unicorn
        Charlie
        Kilo
        """
        self.model = Classifier(classname="weka.classifiers.bayes.net.BayesNetGenerator")
        self.model.options = ['-Q', 'weka.classifiers.bayes.net.search.local.{}'.format(self.algorithm)]
        self.model.build_classifier(train_set)

    def predict(self, test_set):
        """
        Bravo
        Sierra
        """
        results = []
        for _, inst in enumerate(test_set):
            results.append(self.model.classify_instance(inst))
        return results

    def predict_with_class_names(self, test_set):
        results = []
        for _, inst in enumerate(test_set):
            results.append(self.labels[int(self.model.classify_instance(inst))])
        return results
