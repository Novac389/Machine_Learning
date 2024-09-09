import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error, accuracy_score
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
from neural import Neural

'''
Parallelized Grid Search using joblib library.
joblib_progress return a completion bar of the grid search
'''
def evaluate_model_parallel(Neural, param, X, y, n_folds=5, n_repeats=1, n_jobs=-1, verbose=False):
    cv = RepeatedKFold(n_splits=n_folds, n_repeats=n_repeats)
    split_indices = cv.split(X)
    
    if verbose:
        with joblib_progress("kfold", total=n_folds * n_repeats):
            scores = Parallel(n_jobs=n_jobs)(
                delayed(evaluate_fold)(Neural, param, X, y, train_idx, test_idx) 
                for train_idx, test_idx in split_indices
            )
    else:
        scores = Parallel(n_jobs=n_jobs)(
            delayed(evaluate_fold)(Neural, param, X, y, train_idx, test_idx) 
            for train_idx, test_idx in split_indices
        )
    
    return [param, scores]


'''
Evaluate a single fold
'''
def evaluate_fold(Neural,param, X, y, train_idx, test_idx):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    nn = Neural(
        layers=param['layer_structure'],
        epochs=param['epochs'],
        batch_size=param['batch_size'],
        learning_rate=param['learning_rate'],
        hidden_function=param['hidden_function'],
        output_function=param['output_function'],
        init_method=param['init_method'],
        momentum=param['momentum'],
        momentum_schedule = param['momentum_schedule'],
        optimizer = param['optimizer'],
        l2_lambda=param['l2_lambda'],
        early_stopping = False,
        verbose=0
    )
    nn.fit(X_train, y_train)
    y_pred = nn.predict(X_test)
    try:
        mse = mean_squared_error(y_test, y_pred)
    except Exception as e:
        mse = np.nan 
    return mse
    

#Used in plot_compare() to reshape the shortest array with Nan padding
def pad_losses(losses, target_length):
    current_length = len(losses)
    if current_length == target_length:
        return losses
    padding = [np.nan] * (target_length - current_length)
    return losses + padding

'''
Plot the loss curve of 2 NN models, firt plot with trainings and second plot with validations.
The zoom factor helps to zoom on the left-down side of the plot.
'''
def plot_compare(neural1, neural2, title1='Training Losses', title2='Validation Losses', zoom_factor=None, figsize=(8,4)):
    
    max_epochs_train = max(len(neural1.losses["train"]), len(neural2.losses["train"]))
    max_epochs_val = max(len(neural1.losses["validation"]), len(neural2.losses["validation"]))

    # Pad the training losses
    neural1_train_padded = pad_losses(neural1.losses["train"], max_epochs_train)
    neural2_train_padded = pad_losses(neural2.losses["train"], max_epochs_train)

    # Pad the validation losses
    neural1_val_padded = pad_losses(neural1.losses["validation"], max_epochs_val)
    neural2_val_padded = pad_losses(neural2.losses["validation"], max_epochs_val)
    
    data_train = pd.DataFrame({
        'epochs': range(max_epochs_train),
        'NN1_train': neural1_train_padded,
        'NN2_train': neural2_train_padded
    })

    data_train_melted = pd.melt(data_train, ['epochs'], value_name='MSE', var_name='SET')
    
    # Plot training losses
    plt.figure(figsize=figsize)
    sns.lineplot(data=data_train_melted, x='epochs', y='MSE', hue='SET')
    
    if zoom_factor:
        x_max = max_epochs_train - 1
        y_max = max(data_train_melted['MSE'])
        plt.xlim(0, x_max / zoom_factor)
        plt.ylim(0, y_max / zoom_factor)
    
    plt.title(title1)
    plt.show()
    
    data_val = pd.DataFrame({
        'epochs': range(max_epochs_val),
        'NN1_validation': neural1_val_padded,
        'NN2_validation': neural2_val_padded
    })

    data_val_melted = pd.melt(data_val, ['epochs'], value_name='MSE', var_name='SET')
    
    plt.figure(figsize=figsize)
    sns.lineplot(data=data_val_melted, x='epochs', y='MSE', hue='SET')
    
    if zoom_factor:
        x_max = max_epochs_val - 1
        y_max = max(data_val_melted['MSE'])
        plt.xlim(0, x_max / zoom_factor)
        plt.ylim(0, y_max / zoom_factor)
    
    plt.title(title2)
    plt.show()

    
'''
Plot the training and the validation of a user specified measure (MSE or accuracy) of a model.
It is possible to plot in a log scale by setting log_scale=True
'''
def plot_TR_VAL(Neural, title='Line Plot with Legend', zoom_factor=None, curve='MSE', figsize=(6,4), log_scale=False):
    
    if(curve=='MSE'):
        dim = len(Neural.losses["validation"])
        data = pd.DataFrame({
            'epochs': range(dim),
            'train': Neural.losses["train"],
            'validation': Neural.losses["validation"]
        })
    elif(curve=='accuracy'):
        data = pd.DataFrame({
            'epochs': range(len(Neural.accuracy["validation"])),
            'train': Neural.accuracy["train"],
            'validation': Neural.accuracy["validation"]
        })
    else:
        print("must select curve='MSE' or curve='accuracy'")
        return
    
    data_melted = pd.melt(data, ['epochs'], value_name=curve, var_name='SET')
    plt.figure(figsize=figsize)
    sns.lineplot(data=data_melted, x='epochs', y=curve, hue='SET')

    if zoom_factor:
        if(curve=='accuracy'):
            print('Cannot zoom in accuracy curve')
        else:
            x_max = dim - 1
            y_max = max(data_melted['MSE'])
            plt.xlim(0, x_max / zoom_factor)
            plt.ylim(0, y_max / zoom_factor)
            
    if log_scale:
        plt.yscale('log')
            
    plt.title(title)
    plt.show()
    

'''
Given two arrays (predicted and true labels) return the MEE 
'''
def mee(pred, y):
        # Calculate the Euclidean distance for each sample
        distances = np.sqrt(np.sum((y - pred) ** 2, axis=1))
        
        # Calculate the mean
        error = np.mean(distances)
        return error
    
    
'''
Convert the probability vector result of model.predict(X) into a binary vector {0,1} 
'''
def to_binary(X):
    pred = np.where(X > 0.5, 1, 0)
    return pred
    
    
'''
Train and Evaluate model n times 
Calculate and print the mean and std of train and validation set of:
    1. MSE
    2. MEE (if not categorical)
    3. Accuracy (if categorical)
With unpack=True the function returns the values computed instead of printing
'''
def train_and_evaluate(model, X_train, y_train, X_test, y_test, n_iterations=10, unpack=False):
    
    isCat = ((y_test==0) | (y_test==1)).all() #check if we are performing a classification task
    
    train_errors = []
    test_errors = []
    train_mee = []
    test_mee = []
    
    train_accs = []
    test_accs = []
                                                
    for _ in range(n_iterations): #train and evaluate the model n_iterations time
        model.fit(X_train, y_train)
        train_error = mean_squared_error(model.predict(X_train), y_train)
        test_error = mean_squared_error(model.predict(X_test), y_test)
        train_errors.append(train_error)
        test_errors.append(test_error)
        if(isCat):
            train_acc = accuracy_score(y_train, to_binary(model.predict(X_train)))
            test_acc = accuracy_score(y_test, to_binary(model.predict(X_test)))
            train_accs.append(train_acc)
            test_accs.append(test_acc)
        else: #if not classification task, we calculate also the mee
            mee_tr = mee(model.predict(X_train), y_train)
            mee_ts = mee(model.predict(X_test), y_test)
            train_mee.append(mee_tr)
            test_mee.append(mee_ts)
        model.reset()
        
    #compute the mean and standard deviation of the loss on training and test
    mean_train_error = np.mean(train_errors)
    std_train_error = np.std(train_errors)
    mean_test_error = np.mean(test_errors)
    std_test_error = np.std(test_errors)
    
    #compute the mean and standard deviation of the accuracy on training and test
    if(isCat):
        mean_train_acc = np.mean(train_accs)
        std_train_acc = np.std(train_accs)
        mean_test_acc = np.mean(test_accs)
        std_test_acc = np.std(test_accs)
    else: #compute the mean and std of mee
        mean_mee_tr = np.mean(train_mee)
        std_mee_tr = np.std(train_mee)
        mean_mee_ts = np.mean(test_mee)
        std_mee_ts = np.std(test_mee)
        
 
    if(unpack):
        return mean_train_error, std_train_error, mean_test_error, std_test_error
    else:
        print(f"{'Mean Train Loss':<20} {'Mean Test Loss':<20} {'Std Train Loss':<20} {'Std Test Loss':<20}")
        print("-" * 80)
        print(f"{mean_train_error:<20.4f} {mean_test_error:<20.4f} {std_train_error:<20.4f} {std_test_error: <20.4f}")
        if(isCat):
            print("\n")
            print(f"{'Mean Train Accuracy':<20} {'Mean Test Accuracy':<20} {'Std Train Accuracy':<20} {'Std Test Accuracy':<20}")
            print("-" * 80)
            print(f"{mean_train_acc:<20.4f} {mean_test_acc:<20.4f} {std_train_acc:<20.4f} {std_test_acc: <20.4f}")
        else: #if not categorical
            print("\n")
            print(f"{'Mean Train MEE':<20} {'Mean Test MEE':<20} {'Std Train MEE':<20} {'Std Test MEE':<20}")
            print("-" * 80)
            print(f"{ mean_mee_tr:<20.4f} {mean_mee_ts:<20.4f} {std_mee_tr:<20.4f} {std_mee_ts: <20.4f}")


'''
Compare two models by training and evaluate model n times 
Return the mean and std of train and validation set of the two models (MSE)
'''
def compare_experiment(nn, nn2, X_train, y_train, X_test, y_test, n_iterations=10):

    nn_train_mean, nn_train_std, nn_test_mean, nn_test_std = train_and_evaluate(nn, X_train, y_train, X_test, y_test, n_iterations, unpack=True)

    nn2_train_mean, nn2_train_std, nn2_test_mean, nn2_test_std = train_and_evaluate(nn2, X_train, y_train, X_test, y_test, n_iterations, unpack=True)

    print(f"{'Model':<10} {'Mean_Train_Loss':<20} {'Mean_Val_Loss':<20} {'Std_Train_Loss':<20} {'Std_Val_Loss':<20}")
    print("-" * 90)
    print(f"{'NN':<10} {nn_train_mean:<20.4f} {nn_test_mean:<20.4f} {nn_train_std:<20.4f} {nn_test_std:<20.4f}")
    print(f"{'NN2':<10} {nn2_train_mean:<20.4f} {nn2_test_mean:<20.4f} {nn2_train_std:<20.4f} {nn2_test_std:<20.4f}")
    
    
    
'''
Plot the CM (classical momentum) and NAG (nesterov accelerated gradient) results in term of MSE, increasing the momentum term.
In the x-axis: momentum term. In the y-axis: The MSE.
Both train and test are plotted.
momentum_start specify the initial momentum value. trials specify the number of momentum values to test.
'''
def train_and_evaluate_with_momentum(param, X_train, y_train, X_test, y_test, momentum_start=0.8, trials=10):
    
    momentum_values = np.linspace(momentum_start, 0.99, trials) # generate an array of momentum values, from momentum_start to 0.99
    train_errors_classic = []
    test_errors_classic = []
    train_errors_nesterov = []
    test_errors_nesterov = []
    
    for momentum in momentum_values:
        #CLASSIC MOMENTUM
        model_classic = Neural(layers=param['layers'], 
                       epochs=param['epochs'], 
                       learning_rate=param['learning_rate'],
                       batch_size=param['batch_size'],
                       hidden_function=param['hidden_function'],
                       output_function=param['output_function'],
                       init_method='he',
                       optimizer='classic',
                       momentum=momentum,
                       momentum_schedule=False,
                       l2_lambda=0.0,
                       patience=10,
                       early_stopping=True,
                       verbose=0)
        model_classic.fit(X_train, y_train)
        train_errors_classic.append(mean_squared_error(model_classic.predict(X_train), y_train))
        test_errors_classic.append(mean_squared_error(model_classic.predict(X_test), y_test))

        #NESTEROV MOMENTUM
        model_nesterov = Neural(layers=param['layers'], 
                       epochs=param['epochs'], 
                       learning_rate=param['learning_rate'],
                       batch_size=param['batch_size'],
                       hidden_function=param['hidden_function'],
                       output_function=param['output_function'],
                       init_method='he',
                       optimizer='nesterov',
                       momentum=momentum,
                       momentum_schedule=False,
                       l2_lambda=0.0,
                       patience=10,
                       early_stopping=True,
                       verbose=0)
        model_nesterov.fit(X_train, y_train)
        train_errors_nesterov.append(mean_squared_error(model_nesterov.predict(X_train), y_train))
        test_errors_nesterov.append(mean_squared_error(model_nesterov.predict(X_test), y_test))
    #-----end for

    # Plot 1: Training MSE
    plt.figure(figsize=(6, 4))
    plt.plot(momentum_values, train_errors_classic, label='Training MSE Classic', marker='o')
    plt.plot(momentum_values, train_errors_nesterov, label='Training MSE Nesterov', marker='o')
    plt.xlabel('Momentum')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Training MSE vs Momentum')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot 2: Validation MSE
    plt.figure(figsize=(6, 4))
    plt.plot(momentum_values, test_errors_classic, label='Validation MSE Classic', marker='o')
    plt.plot(momentum_values, test_errors_nesterov, label='Validation MSE Nesterov', marker='o')
    plt.xlabel('Momentum')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Validation MSE vs Momentum')
    plt.legend()
    plt.grid(True)
    plt.show()