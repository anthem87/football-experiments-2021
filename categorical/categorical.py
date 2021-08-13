# %%
import numpy as np
import pandas as pd
import pandas.io.sql as sqlio
import os
from os import listdir
import shutil

from CategoricalMerger import Merger
from CategoricalParser import Parser
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical

import kerastuner as kt
from kerastuner.tuners import RandomSearch
from kerastuner.tuners import BayesianOptimization

import matplotlib.pyplot as plt
import plotly.offline as py

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import LabelEncoder

from sqlalchemy import create_engine
import psycopg2

import seaborn as sns

tensorboard = TensorBoard(
    log_dir='./logs',
    histogram_freq=1,
    write_images=True
)


def set_gpu_on():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            print("LOADED GPU")
        except RuntimeError as e:
            # Visible devices must be set at program startup
            print(e)


# %%

def load_parse_and_merge():
    files = []

    for file in listdir('./data'):
        file = './data/' + file
        files.append(file)

    total = len(files)

    shutil.rmtree('./second-step-data/', ignore_errors=True)
    os.mkdir('./second-step-data/')

    for idx, file in enumerate(files):
        ps = Parser()
        print('Parsing file number ' + str(idx + 1) + ' with name ' + file.title() + ' of ' + str(total))
        ps.parse_excel(file)

    files = []

    for file in listdir('./second-step-data'):
        file = './second-step-data/' + file
        files.append(file)

    nm = Merger()
    nm.load_excels(files)


keras_callbacks = [tensorboard, EarlyStopping()]


def save_to_postgres():
    path = './out/out.xlsx'
    games = pd.read_excel(path, engine='openpyxl')
    conn = create_engine('postgresql+psycopg2://postgres:***REMOVED***@localhost:5432/dl-soccer')
    games.to_sql('games', con=conn, if_exists='replace')


def load_games_slq():
    conn = create_engine('postgresql+psycopg2://postgres:***REMOVED***@localhost:5432/dl-soccer')
    sql = "select * from games;"
    return pd.read_sql_query(sql, conn)


# games = load_games_slq()
games = pd.read_excel("./out/out.xlsx")

y = games['Final-Result']

encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
y = to_categorical(y)

games.drop('Final-Result', axis=1, inplace=True)

X = games
# %%

scaler_X = QuantileTransformer()
scaled_X = pd.DataFrame(scaler_X.fit_transform(X), index=X.index, columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.33, random_state=True)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=False)

model_input_shape = X_train.shape[1]
max_model_input_shape = model_input_shape + (32 * 7)


# %%

def tune_model(hp):
    activation = tf.keras.activations.tanh
    drop_out_layer = Dropout(rate=hp.Float('dropout', min_value=0.1, max_value=0.3, default=0.2, step=0.05))
    initial_learning_rate = hp.Float('learning_rate', 0.00001, 0.1, step=10)
    regularizer_rate = hp.Float('regularizer_rate', 0.01, 0.1, step=10)
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=1000,
        decay_rate=hp.Float('decay_rate', 0.5, 1.0, step=10)
    )
    loss = hp.Choice('loss', values=['categorical_crossentropy'])
    metrics = tf.keras.metrics.CategoricalAccuracy()
    optimizer = keras.optimizers.RMSprop(learning_rate=lr_schedule)

    model = keras.Sequential()

    model.add(Dense(model_input_shape, activation=activation))
    # model.add(drop_out_layer)
    for i in range(hp.Int('n_layers', 7, 12)):
        model.add(Dense(
            hp.Int(f'inner_layer_{i}_units', min_value=model_input_shape, max_value=max_model_input_shape, step=32),
            kernel_regularizer=tf.keras.regularizers.l2(regularizer_rate),
            bias_regularizer=tf.keras.regularizers.l2(regularizer_rate),
            activity_regularizer=tf.keras.regularizers.l2(regularizer_rate),
            activation=activation))
        model.add(drop_out_layer)
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer,
                  loss=loss,
                  metrics=metrics)

    return model


def run_model(model, epochs):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_val, y_val), verbose=True,
                        callbacks=keras_callbacks)

    print(history.history.keys())

    history_dict = history.history
    loss_values = history_dict['loss']
    categorical_crossentropy = history_dict['val_categorical_accuracy']
    epochs = range(1, len(loss_values) + 1)

    plt.plot(epochs, categorical_crossentropy, 'r', label='val_categorical_accuracy')
    plt.plot(epochs, loss_values, 'b', label='loss')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    model.evaluate(X_test, y_test, verbose=True)
    model.save('my_model')
    return model


def define_tuners(hypermodel, directory, project_name):
    shutil.rmtree('./out_tuning_bayesian', ignore_errors=True)
    shutil.rmtree('./out_tuning_random_search', ignore_errors=True)
    shutil.rmtree('./out_tuning_hyperband', ignore_errors=True)

    objective = kt.Objective('val_categorical_accuracy', direction='max')

    random_tuner = RandomSearch(
        hypermodel,
        objective=objective,
        max_trials=20,
        executions_per_trial=1,
        directory=directory + '_random_search',
        project_name=project_name
    )

    bayesian_tuner = BayesianOptimization(
        hypermodel,
        objective=objective,
        max_trials=5,
        executions_per_trial=1,
        directory=directory + '_bayesian',
        project_name=project_name
    )

    hyperband_tuner = kt.Hyperband(
        hypermodel,
        objective=objective,
        max_epochs=30,
        hyperband_iterations=2,
        directory=directory + '_hyperband',
        project_name=project_name
    )

    return [random_tuner, bayesian_tuner, hyperband_tuner]


def make_tuning(idx):
    tuners = define_tuners(tune_model, 'out_tuning', 'dl-soccer')
    tuner = tuners[idx]
    tuner.search_space_summary()
    tuner.search(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))
    tuner.results_summary()
    return tuner


def get_best_model():
    best_model = None

    try:
        tuner = make_tuning(1)

        best_model = tuner.get_best_models(num_models=1)[0]

        loss, accuracy = best_model.evaluate(X_test, y_test)
        best_model.summary()
    except:
        print('error while parsing')
    return best_model


def sample_predict(best_model):
    home = pd.read_excel('./prediction-for-result/Lecce-prediction.xlsx')
    away = pd.read_excel('./prediction-for-result/Chievo-prediction.xlsx')

    match = pd.concat([home, away], axis=1)

    match = np.array(match)

    # print(atalanta_vs_spezia)

    scaler_predict = scaler_X
    scaled_predict = scaler_predict.fit_transform(match)

    raw_predictions = best_model.predict(scaled_predict)
    predictions_max = np.argmax(raw_predictions, axis=1)
    prediction_ = encoder.inverse_transform(predictions_max)

    print(raw_predictions)
    print(prediction_)


def load_model():
    return keras.models.load_model('my_model')


def reset_model(model):
    model.reset_states()


# parse_for_prediction()
set_gpu_on()
# load_parse_and_merge()
# model = load_model()
# reset_model(model)
# model.evaluate(X_test, y_test)
# model.summary()
# save_to_postgres()
# load_games_slq()
model = get_best_model()
run_model(model, 100)
model.evaluate(X_test, y_test)
# model.summary()
# sample_predict(model)


for file in listdir('./second-step-data'):
    file = './second-step-data/' + file
    team = pd.read_excel(file, header=None)
    if (len(team.index) - 1) % 4 != 0:
        print(file)

for file in listdir('./second-step-data'):
    file = './second-step-data/' + file
    team = pd.read_excel(file, header=None)
    print(team)
    team.drop(team.tail(3).index, inplace=True)
    print(team)
    break
