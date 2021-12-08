from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from skopt import BayesSearchCV
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle


class Model:
    def __init__(self):
        self.model = RandomForestClassifier(n_jobs=2, random_state=0)
        self.loaded = False
        self.model_name = ""

    def evaluation(self, data):

        # Tree summary and model evaluation metrics
        print('*************** Tree Summary ***************')
        if self.model_name == "RandomForest":
            print('Tree Depth: ', self.model.max_depth)
        print('Classes: ', self.model.classes_)
        if self.model_name == "CART":
            print('Tree Depth: ', self.model.tree_.max_depth)
            print('No. of leaves: ', self.model.tree_.n_leaves)
        print('No. of features: ', self.model.n_features_)
        print('--------------------------------------------------------\n')

        if data.type_of_split == "Training":
            pred_labels_tr = self.model.predict(data.x_train)
            with np.nditer(pred_labels_tr, op_flags=['readwrite']) as it:
                for val in it:
                    if val != 0 and val != 1:
                        if val < 0.5:
                            val[...] = 0
                        else:
                            val[...] = 1
            pred_labels_te = self.model.predict(data.x_test)
            with np.nditer(pred_labels_te, op_flags=['readwrite']) as it:
                for val in it:
                    if val != 0 and val != 1:
                        if val < 0.5:
                            val[...] = 0
                        else:
                            val[...] = 1

            print('*************** Evaluation on Test Data ***************')
            score_te = self.model.score(data.x_test, data.y_test)
            print('Accuracy Score: ', score_te)
            print(classification_report(data.y_test, pred_labels_te))
            print('--------------------------------------------------------\n')

            print('*************** Evaluation on Training Data ***************')
            score_tr = self.model.score(data.x_train, data.y_train)
            print('Accuracy Score: ', score_tr)
            print(classification_report(data.y_train, pred_labels_tr))
            print('--------------------------------------------------------\n')

            print('*************** Confusion matrix ***************')
            print(confusion_matrix(data.y_test, pred_labels_te))
            print('--------------------------------------------------------\n')

    def create_random_forest(self, data):
        # self.model = RandomForestClassifier(n_jobs=2, max_features='sqrt', min_samples_leaf=3, min_samples_split=25,
        #                                     n_estimators=4, max_depth=5)
        self.model = RandomForestClassifier(n_jobs=2, max_features='log2', min_samples_leaf=2, min_samples_split=58,
                                            n_estimators=164, max_depth=4)
        self.model.fit(data.x_train, data.y_train)
        self.model_name = "RandomForest"

        self.loaded = True

        features_importance(self.model.feature_importances_, data.columns)
        self.evaluation(data)

        pickle.dump(self.model, open(self.model_name, 'wb'))

        print("\tCreated {:s}".format(self.model_name))

    def create_gradient_boosting(self, data):

        # self.model = GradientBoostingClassifier(learning_rate=0.2747, max_features='sqrt', min_samples_leaf=2,
        #                                         min_samples_split=6, n_estimators=64)
        self.model = GradientBoostingClassifier(learning_rate=0.10509785877038393, max_features='log2',
                                                min_samples_leaf=17, min_samples_split=47, n_estimators=907,
                                                max_depth=4)

        self.model.fit(data.x_train, data.y_train)
        self.model_name = "GradientBoosting"

        self.loaded = True

        features_importance(self.model.feature_importances_, data.columns)
        self.evaluation(data)

        pickle.dump(self.model, open(self.model_name, 'wb'))

        print("\tCreated {:s}".format(self.model_name))

    def create_cart(self, data):

        model = DecisionTreeClassifier(max_depth=None, min_samples_leaf=2, min_samples_split=4, max_features='log2')

        self.model = model.fit(data.x_train, data.y_train)
        self.model_name = "CART"

        self.loaded = True

        features_importance(model.feature_importances_, data.columns)
        self.evaluation(data)

        pickle.dump(self.model, open(self.model_name, 'wb'))

        print("\tCreated {:s}".format(self.model_name))

    def load_random_forest(self):
        try:
            self.model_name = "RandomForest"

            self.model = pickle.load(open(self.model_name, 'rb'))

            self.loaded = True

            print("\tLoaded {:s}".format(self.model_name))
        except OSError:
            print("Random Forest Model does not exist")

        except BaseException as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

    def load_gradient_boosting(self):
        try:
            self.model_name = "GradientBoosting"

            self.model = pickle.load(open(self.model_name, 'rb'))

            self.loaded = True

            print("\tLoaded {:s}".format(self.model_name))
        except OSError:
            print("Gradient Boosting Model does not exist")

        except BaseException as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

    def load_cart(self):
        try:
            self.model_name = "CART"

            self.model = pickle.load(open(self.model_name, 'rb'))

            self.loaded = True

            print("\tLoaded {:s}".format(self.model_name))
        except OSError:
            print("CART Model does not exist")

        except BaseException as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

    def simulate(self, x_test, y_test, test_copy, type_of_split, att_or_nor_first):

        print("\tUsing {:s}\n".format(self.model_name))

        x_pred = self.model.predict(x_test)

        preds = test_copy.copy()
        preds = preds.assign(Predictions=x_pred, PredictionValues=x_pred)

        scored = pd.DataFrame(index=preds.index)
        scored = scored.assign(Predictions=x_pred, Threshold=0.5)

        if type_of_split == "Training":
            preds_val_temp = x_pred
            with np.nditer(preds_val_temp, op_flags=['readwrite']) as it:
                for val in it:
                    if val != 0 and val != 1:
                        if val < 0.5:
                            val[...] = 0
                        else:
                            val[...] = 1
            print("Accuracy: {:.4f}%".format(metrics.accuracy_score(preds_val_temp, y_test)))

            scored['Actual'] = y_test
            scored.sort_index(inplace=True)
            scored.plot(figsize=(16, 9), style=['b-', 'r-', 'g--'])
        else:
            scored.plot(figsize=(16, 9), style=['b-', 'r-'])
        plt.ylim(-1, 1.5)
        plt.show()

        preds.reset_index(drop=True, inplace=True)

        preds["Predictions"] = preds["Predictions"].astype(str)

        correct = 0
        incorrect = 0
        normal = 0
        attack = 0

        if type_of_split == "Regular":
            if att_or_nor_first == "Normal":
                for index, value in enumerate(x_pred):
                    if value <= 0.5:
                        preds.at[index, "Predictions"] = "Normal"
                        normal = normal + 1
                    elif value > 0.5:
                        preds.at[index, "Predictions"] = "Attack"
                        attack = attack + 1
            else:
                for index, value in enumerate(x_pred):
                    if value >= 0.5:
                        preds.at[index, "Predictions"] = "Normal"
                        normal = normal + 1
                    elif value < 0.5:
                        preds.at[index, "Predictions"] = "Attack"
                        attack = attack + 1
            normal_percent = (100 * normal) / (normal + attack)
            print("All: {:d}, normal: {:d} ({:.4f}%), attack: {:d} ({:.4f}%)".format(
                (normal + attack), normal, normal_percent, attack,
                (100 - normal_percent)))
            print('--------------------------------------------------------\n')

        else:
            if att_or_nor_first == "Normal":
                for index, value in enumerate(x_pred):
                    if value <= 0.5:
                        preds.at[index, "Predictions"] = "Normal"
                        if preds.at[index, 'Normal/Attack'] == "Normal":
                            correct = correct + 1
                        else:
                            incorrect = incorrect + 1
                    elif value > 0.5:
                        preds.at[index, "Predictions"] = "Attack"
                        if preds.at[index, 'Normal/Attack'] == "Attack":
                            correct = correct + 1
                        else:
                            incorrect = incorrect + 1
            else:
                for index, value in enumerate(x_pred):
                    if value >= 0.5:
                        preds.at[index, "Predictions"] = "Normal"
                        if preds.at[index, 'Normal/Attack'] == "Normal":
                            correct = correct + 1
                        else:
                            incorrect = incorrect + 1
                    elif value < 0.5:
                        preds.at[index, "Predictions"] = "Attack"
                        if preds.at[index, 'Normal/Attack'] == "Attack":
                            correct = correct + 1
                        else:
                            incorrect = incorrect + 1
            correct_percent = (100 * correct) / (correct + incorrect)
            print("All: {:d}, correct: {:d} ({:.4f}%), incorrect: {:d} ({:.4f}%)".format
                  ((correct + incorrect), correct, correct_percent, incorrect, (100 - correct_percent)))
            print('--------------------------------------------------------\n')

            with np.nditer(x_pred, op_flags=['readwrite']) as it:
                for val in it:
                    if val != 0 and val != 1:
                        if val < 0.5:
                            val[...] = 0
                        else:
                            val[...] = 1
            print('*************** Confusion matrix ***************')
            print(confusion_matrix(y_test, x_pred))
            print('--------------------------------------------------------\n')

        print('Writing to a file...')
        preds.to_csv("Predictions{:s}_{:s}.csv".format(self.model_name, type_of_split), float_format="%.8f")
        print('Done\n')


def features_importance(feature_importances, col_names):
    feature_imp = pd.Series(feature_importances, index=col_names).sort_values(ascending=False)
    print(feature_imp)

    sns.barplot(x=feature_imp, y=feature_imp.index)

    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title("Visualizing Important Features")
    plt.subplots_adjust(left=0.23)

    plt.show()


def hyperparameters_random_forest(data):

    model = RandomForestClassifier()

    params = dict()
    params['n_estimators'] = (2, 1000.0, 'log-uniform')
    params['min_samples_split'] = (2, 100.0, 'log-uniform')
    params['min_samples_leaf'] = (1, 100.0, 'log-uniform')
    params['max_depth'] = (1, 100.0, 'log-uniform')
    # 0params['criterion'] = ['friedman_mse', 'squared_error', 'mse']
    params['max_features'] = ['log2', 'sqrt']
    # define evaluation
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # define the search
    search = BayesSearchCV(estimator=model, search_spaces=params, n_jobs=-1, cv=cv)
    # perform the search
    search.fit(data.x_train, data.y_train)
    # report the best result
    print(search.best_score_)
    print(search.best_params_)


def hyperparameters_gradient_boosting(data):

    model = GradientBoostingClassifier()

    # define search space
    params = dict()
    params['n_estimators'] = (2, 1000.0, 'log-uniform')
    params['learning_rate'] = (1e-6, 100.0, 'log-uniform')
    params['min_samples_split'] = (2, 100.0, 'log-uniform')
    params['min_samples_leaf'] = (1, 100.0, 'log-uniform')
    params['max_features'] = ['log2', 'sqrt']
    params['max_depth'] = (1, 100.0, 'log-uniform')
    # define evaluation
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # define the search
    search = BayesSearchCV(estimator=model, search_spaces=params, n_jobs=-1, cv=cv)
    # perform the search
    search.fit(data.x_train, data.y_train)
    # report the best result
    print(search.best_score_)
    print(search.best_params_)


def hyperparameters_cart(data):

    model = DecisionTreeClassifier()

    # define search space
    params = dict()
    params['min_samples_split'] = (2, 100.0, 'log-uniform')
    params['min_samples_leaf'] = (1, 100.0, 'log-uniform')
    params['max_depth'] = (1, 100.0, 'log-uniform')
    params['max_features'] = ['log2', 'sqrt']
    # define evaluation
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # define the search
    search = BayesSearchCV(estimator=model, search_spaces=params, n_jobs=-1, cv=cv)
    # perform the search
    search.fit(data.x_train, data.y_train)
    # report the best result
    print(search.best_score_)
    print(search.best_params_)
