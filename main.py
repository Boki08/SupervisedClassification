import model
import data
import sys

if __name__ == '__main__':

    train_file = 'FullTrain.csv'

    test_file = '20191206MixTest.csv'
    #test_file = 'test.csv'

    data_obj = data.Data(train_file, test_file)
    if not data_obj.imported:
        print("One of the Data files does not exist!")
        sys.exit()
    model_obj = model.Model()
    menu = {'1.': "Import Data", '2.': "Create Model", '3.': "Load Model", '4.': "Simulate", '5.': "Search "
                                                                                                   "Hyperparameters",
            '6.': "Exit"}
    modelMenu = {'1.': "Random Forest", '2.': "Gradient Boosting", '3.': "CART", '4.': "Exit"}
    dataMenu = {'1.': "Regular", '2.': "Training", '3.': "Exit"}
    while True:
        print('************************************************')
        print('Menu:')
        options = menu.keys()
        for entry in options:
            print(entry, menu[entry])

        selection = input("Please Select:")
        if selection == '1':
            print('--------------------------------------------------------')
            print("Import Data")
            while True:
                options = dataMenu.keys()
                for entry in options:
                    print(entry, dataMenu[entry])

                selection = input("Please Select:")
                if selection == '1':
                    data_obj.split_data("Regular")
                    break
                elif selection == '2':
                    data_obj.split_data("Training")
                    break
                elif selection == '3':
                    break
                else:
                    print("Unknown Option Selected!")
        elif selection == '2':
            print('--------------------------------------------------------')
            print("Create Model")
            while True:
                options = modelMenu.keys()
                for entry in options:
                    print(entry, modelMenu[entry])

                selection = input("Please Select:")
                if selection == '1':
                    model_obj.create_random_forest(data_obj)
                    break
                elif selection == '2':
                    model_obj.create_gradient_boosting(data_obj)
                    break
                elif selection == '3':
                    model_obj.create_cart(data_obj)
                    break
                elif selection == '4':
                    break
                else:
                    print("Unknown Option Selected!")
        elif selection == '3':
            print('--------------------------------------------------------')
            print("Load Model")
            while True:
                options = modelMenu.keys()
                for entry in options:
                    print(entry, modelMenu[entry])

                selection = input("Please Select:")
                if selection == '1':
                    model_obj.load_random_forest()
                    break
                elif selection == '2':
                    model_obj.load_gradient_boosting()
                    break
                elif selection == '3':
                    model_obj.load_cart()
                    break
                elif selection == '4':
                    break
                else:
                    print("Unknown Option Selected!")
        elif selection == '4':
            print('--------------------------------------------------------')
            print("Simulate")
            if model_obj.loaded:
                model_obj.simulate(data_obj.x_test, data_obj.y_test, data_obj.test_copy, data_obj.type_of_split,
                                   data_obj.train_copy.iloc[0]["Normal/Attack"])
            else:
                print("\tModel is not loaded!")
        elif selection == '5':
            print('--------------------------------------------------------')
            print("Search Hyperparameters")
            while True:
                options = modelMenu.keys()
                for entry in options:
                    print(entry, modelMenu[entry])

                selection = input("Please Select:")
                if selection == '1':
                    model.hyperparameters_random_forest(data_obj)
                    break
                elif selection == '2':
                    model.hyperparameters_gradient_boosting(data_obj)
                    break
                elif selection == '3':
                    model.hyperparameters_cart(data_obj)
                    break
                elif selection == '4':
                    break
                else:
                    print("Unknown Option Selected!")
        elif selection == '6':
            break
        else:
            print("Unknown Option Selected!")
