from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np


class Data:
    def __init__(self, train_file, test_file):

        self.imported_train = True
        self.imported_test = True

        try:
            self.train_data = pd.read_csv(train_file, sep=';')
        except OSError:
            print(train_file + "does not exist")
            self.imported_train = False
            return
        except BaseException as err:
            print(f"Unexpected {err=}, {type(err)=}")
            self.imported_train = False
            raise

        self.train_copy = self.train_data.copy()

        try:
            self.test_data = pd.read_csv(test_file, sep=';')
        except OSError:
            print(test_file + "does not exist")
            self.imported_test = False
            return
        except BaseException as err:
            print(f"Unexpected {err=}, {type(err)=}")
            self.imported_test = False
            raise

        self.test_copy_full = self.test_data.copy()
        self.test_copy = self.test_data.copy()
        self.val_copy = 0

        self.y_train_full = 0
        self.x_test_copy = 0
        self.x_train_copy = 0

        self.x_train = 0
        self.x_test = 0
        self.y_train = 0
        self.y_test = 0
        self.x_val = 0
        self.y_val = 0

        self.type_of_split = "Train"
        #       self.columns = ["Source", "Destination", "Protocol"]

        self.columns = ["Source", "Destination", "Protocol", "Length", "Source Port", "Destination Port"]
        unique_values = []
        object_columns = []

        cols = [i for i in self.train_data.columns if i in self.columns]
        for col in cols:
            if self.train_data[col].isna().values.any():
                self.train_data[col].fillna(-1, inplace=True)
            if self.train_data[col].dtype == "object":
                object_columns.append(col)
                unique_values = [*unique_values, *self.train_data[col].unique()]
            if self.test_data[col].isna().values.any():
                self.test_data[col].fillna(-1, inplace=True)
            if self.test_data[col].dtype == "object":
                unique_values = [*unique_values, *self.test_data[col].unique()]

        label_encoder = LabelEncoder()
        label_encoder.fit(unique_values)
        for key in object_columns:
            self.train_data[key] = label_encoder.transform(self.train_data[key])
            self.test_data[key] = label_encoder.transform(self.test_data[key])

        self.unique_values_normal_attack = self.train_data['Normal/Attack'].unique()

        self.unique_values_normal_attack[[0, np.where(self.unique_values_normal_attack == "Attack")[0][0]]] = \
            self.unique_values_normal_attack[[np.where(self.unique_values_normal_attack == "Attack")[0][0], 0]]

        for idx, val in enumerate(self.unique_values_normal_attack):
            self.train_data['Normal/Attack'] = self.train_data['Normal/Attack'].replace({val: idx})

        self.y_train = self.train_data["Normal/Attack"].to_numpy()

        self.x_train = self.train_data.drop(columns=self.train_data.columns.difference(self.columns))

        self.x_test = self.test_data.drop(columns=self.test_data.columns.difference(self.columns))

        self.x_test_copy = self.x_test.copy()

        self.x_train_copy = self.x_train.copy()

        self.y_train_full = self.y_train

        sc = StandardScaler()
        self.x_train = sc.fit_transform(self.x_train)
        self.x_test = sc.transform(self.x_test)

        print("\tData Imported")
        print("\tx_train: {:s}, y_train:{:s}".format('{}'.format(self.x_train.shape), '{}'.format(self.y_train.shape)))

        if self.imported_test:
            print("\tx_test: {:s}".format('{}'.format(self.x_test.shape)))

    def split_data(self, type_of_split):
        self.type_of_split = type_of_split
        if type_of_split == "Train":
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_train_copy,
                                                                                    self.y_train_full,
                                                                                    test_size=0.4, random_state=0)
            self.x_test, self.x_val, self.y_test, self.y_val = train_test_split(self.x_test,
                                                                                self.y_test,
                                                                                test_size=0.5, random_state=0)
            self.val_copy = self.x_val
            self.val_copy = self.train_copy.loc[self.x_val.index.tolist()]

            sc = StandardScaler()
            self.x_train = sc.fit_transform(self.x_train)
            self.x_test = sc.transform(self.x_test)
            self.x_val = sc.transform(self.x_val)

            print("\tData shape set to train")
            print("\tx_train: {:s}, y_train:{:s}".format('{}'.format(self.x_train.shape),
                                                         '{}'.format(self.y_train.shape)))
            print("\tx_test: {:s}, y_test:{:s}".format('{}'.format(self.x_test.shape), '{}'.format(self.y_test.shape)))
            print("\tx_val: {:s}, y_val:{:s}".format('{}'.format(self.x_val.shape), '{}'.format(self.y_val.shape)))
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
