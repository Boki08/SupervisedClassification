from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from skopt import BayesSearchCV
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pickle


class Model:
    def __init__(self):
        self.model = RandomForestClassifier(n_jobs=2, random_state=0)
        self.loaded = False
        self.model_name = ""

    def evaluation(self, type_of_split, x_train, y_train, x_val, y_val):

        # Tree summary and model evaluation metrics
        print('* {:s} *'.format(self.model_name).center(55))
        print('********************* Tree Summary *********************')
        print('Classes: ', self.model.classes_)
        if self.model_name != "CART":
            print('Tree Depth: ', self.model.max_depth)
        else:
            print('Tree Depth: ', self.model.tree_.max_depth)
            print('No. of leaves: ', self.model.tree_.n_leaves)
        print('No. of features: ', self.model.n_features_)
        print('--------------------------------------------------------\n')

        if type_of_split == "Train":
            pred_labels_train = self.model.predict(x_train)
            with np.nditer(pred_labels_train, op_flags=['readwrite']) as it:
                for val in it:
                    if val != 0 and val != 1:
                        if val < 0.5:
                            val[...] = 0
                        else:
                            val[...] = 1
            pred_labels_test = self.model.predict(x_val)
            with np.nditer(pred_labels_test, op_flags=['readwrite']) as it:
                for val in it:
                    if val != 0 and val != 1:
                        if val < 0.5:
                            val[...] = 0
                        else:
                            val[...] = 1

            print('* {:s} *'.format(self.model_name).center(55))
            print('*************** Evaluation on Validation Data ****************')
            score_te = self.model.score(x_val, y_val)
            print('Accuracy Score: ', score_te)
            print(classification_report(y_val, pred_labels_test))
            print('--------------------------------------------------------\n')

            print('* {:s} *'.format(self.model_name).center(55))
            print('*************** Evaluation on Train Data ***************')
            score_tr = self.model.score(x_train, y_train)
            print('Accuracy Score: ', score_tr)
            print(classification_report(y_train, pred_labels_train))
            print('--------------------------------------------------------\n')

            cf_matrix = confusion_matrix(y_val, pred_labels_test)
            print('* {:s} *'.format(self.model_name).center(55))
            print('******************* Confusion matrix *******************')
            print(cf_matrix)
            print('--------------------------------------------------------\n')

            print_confusion_matrix(cf_matrix, '{:s} (Validation Data)'.format(self.model_name))

    def create_random_forest(self, data):

        self.model = RandomForestClassifier(n_jobs=2, max_features='log2', min_samples_leaf=2, min_samples_split=58,
                                            n_estimators=164, max_depth=4)
        self.model.fit(data.x_train, data.y_train)
        self.model_name = "RandomForest"

        self.loaded = True

        features_importance(self.model.feature_importances_, data.columns, self.model_name)
        self.evaluation(data.type_of_split, data.x_train, data.y_train, data.x_val, data.y_val)

        pickle.dump(self.model, open(self.model_name, 'wb'))

        print("\tCreated {:s}".format(self.model_name).center(48))

    def create_gradient_boosting(self, data):

        self.model = GradientBoostingClassifier(learning_rate=0.105097, max_features='log2', loss='deviance',
                                                min_samples_leaf=17, min_samples_split=47, n_estimators=907,
                                                max_depth=4)

        self.model.fit(data.x_train, data.y_train)
        self.model_name = "GradientBoosting"

        self.loaded = True

        features_importance(self.model.feature_importances_, data.columns, self.model_name)
        self.evaluation(data.type_of_split, data.x_train, data.y_train, data.x_val, data.y_val)

        pickle.dump(self.model, open(self.model_name, 'wb'))

        print("\tCreated {:s}".format(self.model_name).center(48))

    def create_cart(self, data):

        model = DecisionTreeClassifier(max_depth=None, min_samples_leaf=2, min_samples_split=4, max_features='log2')

        self.model = model.fit(data.x_train, data.y_train)
        self.model_name = "CART"

        self.loaded = True

        features_importance(model.feature_importances_, data.columns, self.model_name)
        self.evaluation(data.type_of_split, data.x_train, data.y_train, data.x_val, data.y_val)

        pickle.dump(self.model, open(self.model_name, 'wb'))

        print("\tCreated {:s}".format(self.model_name).center(48))

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

    def simulate(self, x_test, test_copy):  # regular split

        print("\tUsing {:s}\n".format(self.model_name))

        x_pred = self.model.predict(x_test)

        preds = test_copy.copy()
        preds = preds.assign(Predictions=x_pred, PredictionValues=x_pred)

        scored = pd.DataFrame(index=preds.index)
        scored = scored.assign(Predictions=x_pred, Threshold=0.5)

        fig, ax = plt.subplots()

        scored.plot(figsize=(16, 9), style=['b-', 'r-'], ax=ax)
        plt.ylim(-1, 1.5)

        trans = transforms.blended_transform_factory(ax.get_yticklabels()[0].get_transform(), ax.transData)
        ax.text(0.1, -0.25, "{:s}".format("Attack"), color="red", transform=trans, ha="right", va="center")
        ax.text(0.1, 1.25, "{:s}".format("Normal"), color="green", transform=trans, ha="right", va="center")
        plt.show()

        preds.reset_index(drop=True, inplace=True)

        preds["Predictions"] = preds["Predictions"].astype(str)

        normal = 0
        attack = 0


        for index, value in enumerate(x_pred):
            if value >= 0.5:
                preds.at[index, "Predictions"] = "Normal"
                normal += 1
            elif value < 0.5:
                preds.at[index, "Predictions"] = "Attack"
                attack += 1

        normal_percent = (100 * normal) / (normal + attack)
        print("All: {:d}, normal: {:d} ({:.4f}%), attack: {:d} ({:.4f}%)".format(
            (normal + attack), normal, normal_percent, attack,
            (100 - normal_percent)))
        print('--------------------------------------------------------\n')

        print('Writing to a file...')
        preds.to_csv("Predictions{:s}_Regular.csv".format(self.model_name), float_format="%.8f")
        print('Done\n')

    def simulate_val(self, x_test, y_test, test_copy):  # validation split

        print("\tUsing {:s}\n".format(self.model_name))

        x_pred = self.model.predict(x_test)

        preds = test_copy.copy()
        preds = preds.assign(Predictions=x_pred, PredictionValues=x_pred)

        preds_val_temp = x_pred
        with np.nditer(preds_val_temp, op_flags=['readwrite']) as it:
            for val in it:
                if val != 0 and val != 1:
                    if val < 0.5:
                        val[...] = 0
                    else:
                        val[...] = 1
        print("Accuracy: {:.4f}%".format(accuracy_score(preds_val_temp, y_test)))

        cf_matrix = confusion_matrix(y_test, x_pred)
        print('******************* Confusion matrix *******************')
        print(cf_matrix)
        print('--------------------------------------------------------\n')

        print_confusion_matrix(cf_matrix, '{:s} (Test Data)'.format(self.model_name))

        preds.reset_index(drop=True, inplace=True)

        preds["Predictions"] = preds["Predictions"].astype(str)

        true_attack = cf_matrix[0, 0]
        true_normal = cf_matrix[1, 1]
        false_attack = cf_matrix[1, 0]
        false_normal = cf_matrix[0, 1]


        for index, value in enumerate(x_pred):
            if value >= 0.5:
                preds.at[index, "Predictions"] = "Normal"
            elif value < 0.5:
                preds.at[index, "Predictions"] = "Attack"

        correct = true_normal + true_attack
        incorrect = false_normal + false_attack
        correct_percent = (100 * correct) / (correct + incorrect)

        accuracy = correct / (correct + incorrect)
        sensitivity = true_attack / (false_normal + true_attack)
        precision = true_attack / (true_attack + false_attack)
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)

        print('* {:s} *'.format(self.model_name).center(55))
        print('**************** Evaluation on Test Data ****************')
        print("All: {:d}, correct: {:d} ({:.4f}%), incorrect: {:d} ({:.4f}%)\nTesting Accuracy: {:.4f}\nSensitivity("
              "Recall): {:.4f}\nPrecision: {:.4f}\nF1-Score: {:.4f}".format
              ((correct + incorrect), correct, correct_percent, incorrect, (100 - correct_percent), accuracy,
               sensitivity, precision, f1_score))
        print('---------------------------------------------------------------\n')

        print('Writing to a file...')
        preds.to_csv("Predictions{:s}_Test.csv".format(self.model_name), float_format="%.8f")
        print('Done\n')


def print_confusion_matrix(cf_matrix, model_name):
    group_names = ["True Attack", "False Normal", "False Attack", "True Normal"]
    group_counts = ["{0:0.0f}".format(value) for value in
                    cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                         cf_matrix.flatten() / np.sum(cf_matrix)]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')

    ax.set_title("Confusion Matrix - {:s}\n\n".format(model_name))
    ax.set_xlabel("Predicted Values")
    ax.set_ylabel("Actual Values")

    ax.xaxis.set_ticklabels(["Attack", "Normal"])
    ax.yaxis.set_ticklabels(["Attack", "Normal"])

    fig = plt.gcf()
    fig.subplots_adjust(left=0.13, right=0.85, bottom=0.13, top=0.85)
    plt.show()


def features_importance(feature_importances_, col_names, model_name):
    feature_imp = pd.Series(feature_importances_, index=col_names).sort_values(ascending=True)
    print(feature_imp)

    fig, ax = plt.subplots()
    ax = feature_imp.plot(kind="barh", color=plt.rcParams['axes.prop_cycle'].by_key()['color'])
    ax.bar_label(ax.containers[0], fmt="%.4f")
    ax.set_xlabel('Feature Importance Score')
    ax.set_ylabel('Features')
    ax.set_title("Visualizing Features Importance - {:s}".format(model_name))
    fig.subplots_adjust(left=0.23)
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, 1.2 * xmax)
    plt.show()


def hyperparameters_random_forest(x_train, y_train):
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
    search.fit(x_train, y_train)
    # report the best result
    print(search.best_score_)
    print(search.best_params_)


def hyperparameters_gradient_boosting(x_train, y_train):
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
    search.fit(x_train, y_train)
    # report the best result
    print(search.best_score_)
    print(search.best_params_)


def hyperparameters_cart(x_train, y_train):
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
    search.fit(x_train, y_train)
    # report the best result
    print(search.best_score_)
    print(search.best_params_)
