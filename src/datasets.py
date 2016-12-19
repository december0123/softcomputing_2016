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


class LearningSetFactory(object):

    def __init__(self, test_size=0.2):
        print(test_size)
        self.test_size = test_size

    @unique
    class DataSource(Enum):
        breast_cancer = 1
        activity_recognition = 2

    def get_train_test_data(self, data_source):
        print("Data source: {}" .format(data_source))
        if data_source == LearningSetFactory.DataSource.breast_cancer:
            data, target, feature_names = self.get_breast_cancer()
        elif data_source == LearningSetFactory.DataSource.activity_recognition:
            return self.get_activity_recognition()
        else:
            raise Exception("Invalid data source. Check DataSource enum class.")

        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=self.test_size)
        return X_train, y_train, [len(y_train)], X_test, y_test, [len(y_test)], feature_names

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
