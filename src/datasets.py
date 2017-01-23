from collections import defaultdict
import csv
import os
import zipfile
import random
import urllib.request
import numpy as np
from src.settings import DATASET_DIR
from enum import Enum, unique
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from src.markov_chain_transition_matrix_generator import MarkovChainTransitionMatrixGenerator
import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.filters import Filter


class LearningSetFactory(object):
    def __init__(self, test_size=0.2, sequence_length=2):
        print(test_size)
        self.test_size = test_size
        self.sequence_length = sequence_length

    @unique
    class DataSource(Enum):
        breast_cancer = 1
        activity_recognition = 2
        sequenced_breast_cancer = 3
        sequenced_activity_recognition = 4
        weka_breast_cancer = 5

    def get_train_test_data(self, data_source):
        print("Data source: {}" .format(data_source))
        if data_source == LearningSetFactory.DataSource.breast_cancer:
            data, target, feature_names = self.get_breast_cancer()
        elif data_source == LearningSetFactory.DataSource.activity_recognition:
            return self.get_activity_recognition()
        elif data_source == LearningSetFactory.DataSource.sequenced_breast_cancer:
            return self.get_sequenced_breast_cancer()
        elif data_source == LearningSetFactory.DataSource.sequenced_activity_recognition:
            return self.get_sequenced_activity_recognition()
        elif data_source == LearningSetFactory.DataSource.weka_breast_cancer:
            return self.get_weka_breast_cancer()
        else:
            raise Exception("Invalid data source. Check DataSource enum class.")

        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=self.test_size)
        return X_train, y_train, [len(y_train)], X_test, y_test, [len(y_test)], feature_names

    def create_split_sequenced_data(self, features, grouped_by_labels, labels):
        sequenced_X, sequenced_Y = self.create_sequences(grouped_by_labels, labels)
        sequenced_X, sequenced_Y = self.shuffle_sequences(sequenced_X, sequenced_Y)
        if isinstance(sequenced_X[0], dict):
            vectorizer = DictVectorizer(sparse=False)
            sequenced_X = vectorizer.fit_transform(sequenced_X)
        X_test, X_train, y_test, y_train = self.split_sets(sequenced_X, sequenced_Y)
        train_seq_lengths = self.get_sequence_lengths(y_train)
        test_seq_lengths = self.get_sequence_lengths(y_test)
        return X_train, y_train, train_seq_lengths, X_test, y_test, test_seq_lengths, features

    def get_sequenced_breast_cancer(self):
        # proof of concept
        grouped_by_labels, labels, features = self.extract_breast_cancer_data_grouped_by_label()
        return self.create_split_sequenced_data(features, grouped_by_labels, labels)

    def get_sequenced_activity_recognition(self):
        # WARNING
        # this dataset is HUGE
        X_train, y_train, sequence_length_train, X_test, y_test, sequence_length_test, features = self.get_activity_recognition()
        samples = np.concatenate((X_train, X_test))
        labels = np.concatenate((y_train, y_test))
        grouped_by_labels = defaultdict(list)
        for label, sample in zip(labels, samples):
            grouped_by_labels[label].append(list(sample))

        return self.create_split_sequenced_data(features, grouped_by_labels, labels)

    def create_sequences(self, grouped_by_labels, labels):
        # the idea behind this is to make sequenced data out of non sequenced data:
        # 1. create a matrix with probabilities to switch class or stay
        # 2. sort samples by class
        # 3. for all samples
        # 4.    pick the next class with respect to the probabilities in the matrix
        # 5.    pick a random sample from that class
        # 6.    we have a sequence!

        # 1.
        gen = MarkovChainTransitionMatrixGenerator(len(grouped_by_labels))
        # 2.
        matrix = gen.get_transition_probability_matrix()
        # matrix = np.asarray([[0.7, 0.3], [0.3, 0.7]])
        current_label_index = 0
        sequenced_X = []
        sequenced_Y = []

        # activity dataset is HUGE and if statement or try/except in
        # the loop body would be too slow
        def on_dict(grouped_by_labels, next_label):
            return np.random.choice(grouped_by_labels[next_label])

        def on_list(grouped_by_labels, next_label):
            index = np.random.randint(0, len(grouped_by_labels[next_label]))
            return grouped_by_labels[next_label][index]
        if isinstance(list(grouped_by_labels.values())[0][0], dict):
            get_next_in_seq = on_dict
        else:
            get_next_in_seq = on_list
        # 3.
        # note: dicts are not ordered but chyba wyjebane
        for label, rows in grouped_by_labels.items():
            for row in rows:
                sequenced_X.append(row)
                sequenced_Y.append(label)
                # 4.
                # False is kinda zero, so this works
                possible_next_labels_indices = np.nonzero(matrix[current_label_index] > 0)[0]
                probabilities = matrix[current_label_index][possible_next_labels_indices]
                next_label_index = np.random.choice(possible_next_labels_indices, p=probabilities)
                next_label = labels[next_label_index]
                # 5.
                next_in_seq = get_next_in_seq(grouped_by_labels, next_label)
                # 6.
                sequenced_X.append(next_in_seq)
                sequenced_Y.append(next_label)
        return sequenced_X, sequenced_Y

    # MultinomialHMM requires an array containing lengths of all sequences
    def get_sequence_lengths(self, y_train):
        return [self.sequence_length for _ in range(int(len(y_train) / self.sequence_length))]

    def extract_breast_cancer_data_grouped_by_label(self):
        labels = []
        grouped = defaultdict(list)
        bool_map = {'yes': 1, 'no': 0, 'nan': 0}
        with open(os.path.join(DATASET_DIR, 'uci-20070111-breast-cancer.csv')) as csvfile:
            readCSV = csv.DictReader(csvfile)
            for row in readCSV:
                for key in row:
                    row[key] = row[key].strip("'")
                row['deg-malig'] = int(row['deg-malig'])
                row['node-caps'] = bool_map[row['node-caps']]
                row['irradiat'] = bool_map[row['irradiat']]
                label = row['label']
                labels.append(label)
                row.pop('label')
                grouped[label].append(row)
        features = ('age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'breast-quad',
         'irradiat')
        return grouped, labels, features

    def get_split_point(self, sequenced_Y):
        split_point = int(0.2 * len(sequenced_Y))
        while split_point % self.sequence_length > 0:
            split_point += 1
        return split_point

    def split_sets(self, sequenced_X, sequenced_Y):
        split_point = self.get_split_point(sequenced_Y=sequenced_Y)
        X_test = sequenced_X[:split_point]
        X_train = sequenced_X[split_point:]
        y_test = sequenced_Y[:split_point]
        y_train = sequenced_Y[split_point:]
        return X_test, X_train, y_test, y_train

    def shuffle_sequences(self, sequenced_X, sequenced_Y, seq_len=2):
        seq_length_counter = 0
        to_shuffle = []
        package = []
        for zipped in zip(sequenced_X, sequenced_Y):
            package.append(zipped)
            seq_length_counter += 1
            if seq_length_counter == seq_len:
                to_shuffle.append(package)
                seq_length_counter = 0
                package = []
        random.shuffle(to_shuffle)
        sequenced_X = []
        sequenced_Y = []
        for list_of_sequences in to_shuffle:
            for zipped_sequence in list_of_sequences:
                sequenced_X.append(zipped_sequence[0])
                sequenced_Y.append(zipped_sequence[1])
        return sequenced_X, sequenced_Y

    def get_weka_breast_cancer(self):
        split_ratio = 0.2

        loader = Loader(classname="weka.core.converters.CSVLoader")
        loader.options = ['-F', ',']
        dataset = loader.load_file(os.path.join(DATASET_DIR, 'uci-20070111-breast-cancer.csv'))
        dataset.class_is_last()
        remove = Filter(classname="weka.filters.unsupervised.instance.RemovePercentage", options=[
            "-P", str(split_ratio * 100)])
        remove.inputformat(dataset)
        train_set = remove.filter(dataset)
        remove = Filter(classname="weka.filters.unsupervised.instance.RemovePercentage", options=[
            "-P", str(split_ratio * 100), "-V"])
        remove.inputformat(dataset)
        test_set = remove.filter(dataset)

        labels = dataset.class_attribute.values

        return train_set, test_set, labels

    def get_breast_cancer(self):
        data = []
        target = []
        bool_map = {'yes': 1, 'no': 0, 'nan': 0}
        with open(os.path.join(DATASET_DIR, 'uci-20070111-breast-cancer.csv')) as csvfile:
            readCSV = csv.DictReader(csvfile)
            for row in readCSV:
                for key in row:
                    row[key] = row[key].strip("'")
                row['deg-malig'] = int(row['deg-malig'])
                row['node-caps'] = bool_map[row['node-caps']]
                row['irradiat'] = bool_map[row['irradiat']]
                target.append(row['label'])
                row.pop('label')
                data.append(row)

        vectorizer = DictVectorizer(sparse=False)
        data = vectorizer.fit_transform(data, target)
        return data, target, vectorizer.get_feature_names()

    def get_activity_recognition(self):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/" \
              "00287/Activity%20Recognition%20from%20Single%20Chest-Mounted%20Accelerometer.zip"

        target_path = os.path.join(DATASET_DIR, "ARFSCMA.zip")

        if not os.path.exists(target_path):
            urllib.request.urlretrieve(url, target_path)

        unpack_dir = os.path.join(DATASET_DIR, "Activity Recognition from Single Chest-Mounted Accelerometer")

        if not os.path.exists(unpack_dir):
            with zipfile.ZipFile(target_path, "r") as zip_ref:
                zip_ref.extractall(os.path.join(DATASET_DIR))

        csv_files = [os.path.join(unpack_dir, file) for file in os.listdir(unpack_dir) if file.endswith(".csv")]
        random.shuffle(csv_files)

        split_point = int(0.2 * len(csv_files))
        test = slice(0, split_point)
        train = slice(split_point, len(csv_files))

        feature_names = ['x acceleration', 'y acceleration', 'z acceleration']
        return self.get_data_from_csv_without_header(csv_files[test], skip_columns=1) + \
               self.get_data_from_csv_without_header(csv_files[train], skip_columns=1) + (feature_names,)

    def get_data_from_csv_without_header(self, filenames, skip_columns=0):
        data = []
        labels = []
        sequence_lengths = []
        for file in filenames:
            counter = 0
            with open(file, 'r') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    *raw_data, label = row
                    raw_data = raw_data[skip_columns:]
                    data.append(raw_data)
                    labels.append(label)
                    counter += 1

            sequence_lengths.append(counter)

        return np.asarray(data, np.float64), np.asarray(labels), sequence_lengths
