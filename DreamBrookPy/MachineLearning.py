### General Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import pickle
import math


### Machine Learning Import
from keras.models import Sequential 
from keras.layers import Dense, LSTM, Dropout 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

### Task Specific Imports
from metar_taf_parser.parser.parser import MetarParser


############ DATA PROCESSING #############################

def extract_data(the_dataset, keyword, output_list,data_index):
    """
    Extract specific information from the full metar dataset based on certain keywords
    """
    error_data = [] # just to check that the length is the same as what the original are. 
    for i in data_index:
        try:
            dat = the_dataset[i]
            new_data = dat[keyword]
            output_list.append(new_data)
        except Exception as e:
             error_data.append([i,e])
    print(f'length of the error list is : {len(error_data)}')
    return output_list

############ MACHINE LEARNING #############################
def prepare_training_data(training_set_scaled, window_size):
    """
    Prepares training data for time series forecasting models.

    This function takes a scaled training dataset and a specified window size to create
    sequences of data points (features) and their corresponding future value (label) for
    training time series forecasting models. Each sequence of data points (x_train) is
    used along with the next data point (y_train) to create a supervised learning dataset.

    Parameters:
    - training_set_scaled (numpy.ndarray): A 2D numpy array of scaled training data,
      where each row represents a time step and each column represents a feature.
    - window_size (int): The number of time steps to include in each input sequence, i.e.,
      the size of the window of past observations that will be used to predict the next value.

    Returns:
    - x_train (numpy.ndarray): A 3D numpy array of shape (number of samples, window_size, number of features),
      containing the input sequences for training.
    - y_train (numpy.ndarray): A 1D numpy array containing the corresponding labels for each input sequence,
      i.e., the value at the next time step.

    Usage:
    >>> training_set_scaled = np.random.rand(100, 1)  # Example scaled training data
    >>> window_size = 5
    >>> x_train, y_train = prepare_training_data(training_set_scaled, window_size)
    >>> print(x_train.shape, y_train.shape)
    """
    x_train = []
    y_train = []

    data_length = len(training_set_scaled)
    for i in range(window_size, data_length):
        x_train.append(training_set_scaled[i-window_size:i, 0])  # Input sequence
        y_train.append(training_set_scaled[i, 0])  # Corresponding label

    # Convert lists to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)
    # Reshape x_train into 3D array for model input
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    return x_train, y_train


# Build LSTM Cell
def build_LSTM(x_train, y_train, n_units, dropout_rate, n_dense, n_epochs, n_batch, LSTM_depth,
               opt_type,loss_type):
    """

    Usage:
    build_LSTM(x_train,n_units, dropout_rate, n_dense, n_epochs, n_batch, LSTM_depth,
               opt_type,loss_type)


    Typical Values
    n_units = 50
    dropout_rate = 0.2
    n_dense = 1
    n_epochs = 100
    n_batch = 32
    opt_type = 'adam'
    loss_type = 'mean_squared_error'
    """
    # Constructing our LSTM Layer
    
    #initialize NN
    regressor = Sequential()
    
    # 50 is dimensionality of output space, True is return last output (full sequence), 0.2 = 20% dropped randomly
    # Dropout help with minimizing overfitting
    regressor.add(LSTM(units=n_units, return_sequences = True, input_shape = (x_train.shape[1],1)))
    regressor.add(Dropout(dropout_rate))

    # Stack more layers [Check influence of LSTM layer depth]
    if LSTM_depth == 2:
        regressor.add(LSTM(units = n_units, return_sequences = True))
        regressor.add(Dropout(dropout_rate))
    
    elif LSTM_depth == 3:
        regressor.add(LSTM(units = n_units, return_sequences = True))
        regressor.add(Dropout(dropout_rate))
        
        regressor.add(LSTM(units = n_units, return_sequences = True))
        regressor.add(Dropout(dropout_rate))
    
    elif LSTM_depth == 4:
        regressor.add(LSTM(units = n_units, return_sequences = True))
        regressor.add(Dropout(dropout_rate))
    
        regressor.add(LSTM(units = n_units, return_sequences = True))
        regressor.add(Dropout(dropout_rate))
        
        regressor.add(LSTM(units = n_units, return_sequences = True))
        regressor.add(Dropout(dropout_rate))
    
    else: 
        print("please specify LSTM depth between 2-4")
    #no need to return sequence here because only final output is needed for Dense Layer
    regressor.add(LSTM(units = n_units))
    regressor.add(Dropout(dropout_rate))
    
    #Dense Layer for regression prediction
    regressor.add(Dense(units = n_dense))
    
    # Specifying Loss Function for optimization
    regressor.compile(optimizer = opt_type, loss = loss_type)
    
    # Define batch size and training durations.
    regressor.fit(x_train, y_train, epochs = n_epochs, batch_size = n_batch)

    return regressor


def prepare_test_data(inputs, scaling, window_size, test_data_length):
    """
    Prepare test data for time series prediction models.

    This function scales the input data, segments it into a sequence of data points
    based on the specified window size, and reshapes it for compatibility with
    time series forecasting models.

    Parameters:
    - inputs (numpy.ndarray): The input data to be processed, expected to be a 2D array
      where rows correspond to data points and columns to features.
    - scaling (object): A fitted scaler object (e.g., from scikit-learn) used to
      normalize or standardize the inputs. Must have a `transform` method.
    - window_size (int): The number of consecutive data points used to create each
      test data sample.
    - test_data_length (int): The number of test samples to generate from the inputs.

    Returns:
    - numpy.ndarray: A 3D array of shape (test_data_length, window_size, 1),
      ready for use with time series prediction models. Each test sample is a
      sequence of `window_size` consecutive data points from the scaled input data,
      reshaped to meet the input requirements of typical forecasting models.

    Example usage:
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> import numpy as np
    >>> inputs = np.random.rand(100, 1)  # Example input data
    >>> scaler = MinMaxScaler().fit(inputs)  # Fit a scaler to the inputs
    >>> window_size = 5
    >>> test_data_length = 10
    >>> x_test = prepare_test_data(inputs, scaler, window_size, test_data_length)
    >>> print(x_test.shape)
    (10, 5, 1)
    """
    inputs = scaling.transform(inputs)
    x_test = []
    for i in range(window_size,window_size + test_data_length):
        x_test.append(inputs[i-window_size:i,0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
    return x_test

def get_prediction(my_regressor, x_test, scaling):   
    """
    Docstring here AI will do the job later
    """
    prediction = my_regressor.predict(x_test)
    prediction = scaling.inverse_transform(prediction)
    return prediction

#Evaluate Performance
def return_rmse(test,predicted):
    """
    Docstring here AI will do the job later
    """
    rmse = math.sqrt(mean_squared_error(test,predicted))
    print(f"The root mean squared error is {rmse}.")
    return rmse