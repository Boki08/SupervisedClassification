from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


class Data:
    def __init__(self, train_file, test_file):

        self.imported = True

        try:
            self.train_data = pd.read_csv(train_file, sep=';')
        except OSError:
            print(train_file + "does not exist")
            self.imported = False
            return
        except BaseException as err:
            print(f"Unexpected {err=}, {type(err)=}")
            self.imported = False
            raise

        self.train_copy = self.train_data.copy()

        try:
            self.test_data = pd.read_csv(test_file, sep=';')
        except OSError:
            print(test_file + "does not exist")
            self.imported = False
            return
        except BaseException as err:
            print(f"Unexpected {err=}, {type(err)=}")
            self.imported = False
            raise

        self.test_copy_full = self.test_data.copy()
        self.test_copy = self.test_data.copy()

        self.y_train_full = 0
        self.x_test_copy = 0
        self.x_train_copy = 0

        self.x_train = 0
        self.x_test = 0
        self.y_train = 0
        self.y_test = 0

        self.type_of_split = "Regular"

        self.columns = ["Source", "Destination", "Protocol", "Length", "Source Port", "Destination Port"]

        #       self.columns = ["Source", "Destination", "Protocol"]

        unique_values_train = {}
        unique_values_test = {}
        unique_values = {}
        cols = [i for i in self.train_data.columns if i in self.columns]
        for col in cols:
            if self.train_data[col].isna().values.any():
                self.train_data[col].fillna(-1, inplace=True)
            if self.train_data[col].dtype == "object":
                unique_values_train[col] = self.train_data[col].unique()
            if self.test_data[col].isna().values.any():
                self.test_data[col].fillna(-1, inplace=True)
            if self.test_data[col].dtype == "object":
                unique_values_test[col] = self.test_data[col].unique()

        self.unique_values_normal_attack = self.train_data['Normal/Attack'].unique()

        for idx, val in enumerate(self.unique_values_normal_attack):
            self.train_data['Normal/Attack'] = self.train_data['Normal/Attack'].replace({val: idx})

        for key in unique_values_train:
            unique_values[key] = np.hstack([unique_values_train[key], unique_values_test[key]])
            unique_values[key] = unique_values[key][np.sort(np.unique(unique_values[key], return_index=True)[1])]
            for idx, val in enumerate(unique_values[key]):
                if val in unique_values_train[key]:
                    self.train_data = self.train_data.replace({val: idx})
                if val in unique_values_test[key]:
                    self.test_data = self.test_data.replace({val: idx})

        self.y_train = pd.factorize(self.train_data["Normal/Attack"])[0]

        self.x_train = self.train_data.drop(columns=self.train_data.columns.difference(self.columns))

        self.x_test = self.test_data.drop(columns=self.test_data.columns.difference(self.columns))

        self.x_test_copy = self.x_test.copy()

        self.x_train_copy = self.x_train.copy()

        self.y_train_full = self.y_train

        sc = StandardScaler()
        self.x_train = sc.fit_transform(self.x_train)
        self.x_test = sc.transform(self.x_test)

        print("\tData set shape to regular")
        print("\tx_train: {:s}, y_train:{:s}".format('{}'.format(self.x_train.shape), '{}'.format(self.y_train.shape)))

    def split_data(self, type_of_split):
        self.type_of_split = type_of_split
        if type_of_split == "Training":
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_train_copy,
                                                                                    self.y_train_full,
                                                                                    test_size=0.2, random_state=0)

            self.test_copy = self.x_test
            self.test_copy = self.train_copy.loc[self.x_test.index.tolist()]

            sc = StandardScaler()
            self.x_train = sc.fit_transform(self.x_train)
            self.x_test = sc.transform(self.x_test)
            print("\tData shape set to training")
            print("\tx_train: {:s}, y_train:{:s}".format('{}'.format(self.x_train.shape),
                                                         '{}'.format(self.y_train.shape)))
            print("\tx_test: {:s}, y_test:{:s}".format('{}'.format(self.x_test.shape), '{}'.format(self.y_test.shape)))
        else:
            self.x_train = self.x_train_copy.copy()
            self.y_train = self.y_train_full.copy()
            self.test_copy = self.test_copy_full.copy()
            self.x_test = self.x_test_copy.copy()

            sc = StandardScaler()
            self.x_train = sc.fit_transform(self.x_train)
            self.x_test = sc.transform(self.x_test)
            print("\tData shape set to regular")
            print("\tx_train: {:s}, y_train:{:s}".format('{}'.format(self.x_train.shape),
                                                         '{}'.format(self.y_train.shape)))
