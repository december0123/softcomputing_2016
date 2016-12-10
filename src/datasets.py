import csv
import os
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

    def get_train_test_data(self, data_source):
        if data_source == LearningSetFactory.DataSource.breast_cancer:
            data, target = self.get_breast_cancer()
        else:
            raise Exception("Invalid data source. Check DataSource enum class.")

        return train_test_split(data, target, test_size=self.test_size)

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

        return DictVectorizer(sparse=False).fit_transform(data, target), target
