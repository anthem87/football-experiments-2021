from os import listdir
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import LabelEncoder as encoder

from sqlalchemy import create_engine

from RecurrentMerger import RecurrentMerger
from RecurrentParser import RecurrentParser

# py.init_notebook_mode(connected=True)

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
            tf.config.experimental.set_memory_growth(gpus[0], True)
            tf.config.experimental.set_virtual_device_configuration(gpus[0], [
                tf.config.experimental.VirtualDeviceConfiguration(memory_limit=25.600)])
            print('OK, gpu loaded')
        except RuntimeError as e:
            # Visible devices must be set at program startup
            print(e)


def save_to_postgres(recurrent: bool):
    if recurrent:
        path = './out-recurrent/out.xlsx'
    else:
        path = './out/out.xlsx'

    print('Opening file...')
    games = pd.read_excel(path, engine='openpyxl')
    print('Connecting...')
    conn = create_engine('postgresql+psycopg2://postgres:***REMOVED***@localhost:5432/dl-soccer')
    print('Saving to db...')
    games.to_sql('games_for_recurrent', con=conn, if_exists='replace')


def load_games_slq():
    conn = create_engine('postgresql+psycopg2://postgres:***REMOVED***@localhost:5432/dl-soccer')
    sql = "select * from games_for_recurrent;"
    return pd.read_sql_query(sql, conn)


def load_parse_and_merge():
    files = []

    for file in listdir('./data'):
        file = './data/' + file
        files.append(file)

    total = len(files)

    for idx, file in enumerate(files):
        ps = RecurrentParser()
        print('Parsing file number ' + str(idx + 1) + ' with name ' + file.title() + ' of ' + str(total))
        ps.parse_excel(file)

    files = []

    for file in listdir('./second-step-data-recurrent'):
        file = './second-step-data-recurrent/' + file
        files.append(file)

    nm = RecurrentMerger()
    nm.load_excels(files)


# Define the prior weight distribution as Normal of mean=0 and stddev=1.
# Note that, in this example, the we prior distribution is not trainable,
# as we fix its parameters.


keras_callbacks = [tensorboard, EarlyStopping()]

games = pd.read_excel("./recurrent-out/recurrent-out.xlsx")

games.drop('Final-Result', axis=1, inplace=True)
games.sort_index(ascending=False)

y = []
X = []


def split_input_outputs(data_X: pd.DataFrame):
    stopper = 0
    for idx, row in data_X.iterrows():
        if idx == 4 - 1 + stopper:
            stopper += 4
            y.append(row)
        else:
            X.append(row)
        print('working on row number ' + str(idx))


split_input_outputs(games)

scaler_X = QuantileTransformer()
scaled_X = pd.DataFrame(scaler_X.fit_transform(X))
scaled_y = pd.DataFrame(scaler_X.fit_transform(y))

X_train, y_train = np.array(scaled_X), np.array(scaled_y)

X_train = np.reshape(X_train, (y_train.shape[0], 3, X_train.shape[1]))
y_train = np.reshape(y_train, (y_train.shape[0], 1, y_train.shape[1]))
print(X_train.shape)
print(y_train.shape)

X_train, X_test = train_test_split(X_train, test_size=0.33, random_state=False)
y_train, y_test = train_test_split(y_train, test_size=0.33, random_state=False)

X_train, X_val = train_test_split(X_train, test_size=0.25, random_state=False)
y_train, y_val = train_test_split(y_train, test_size=0.25, random_state=False)

model_input_shape = (X_train.shape[0], X_train.shape[1])
output_shape = y_train.shape[1]


def get_model():
    activation = tf.keras.activations.tanh
    units = 3
    drop_out_layer = Dropout(0.3)
    learning_rate = 0.00001
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        learning_rate,
        decay_steps=10000,
        decay_rate=0.5)
    loss = 'mse'
    metrics = 'mse'
    optimizer = keras.optimizers.RMSprop(learning_rate=lr_schedule)

    model = tf.keras.Sequential()

    model.add(
        Bidirectional(
            LSTM(units=units, return_sequences=True, input_shape=model_input_shape, name='Bidirectional')
        )
    )
    # model.add(drop_out_layer)
    for i in range(7, 12):
        model.add(LSTM(units=units, return_sequences=True, name='LSTM_' + str(i)))

        model.add(drop_out_layer)
    model.add(Dense(output_shape, activation='linear', name='Dense_Output'))

    model.compile(optimizer,
                  loss=loss,
                  metrics=metrics)

    return model


def run_model(model, epochs):
    history = model.fit(X_train, y_train, epochs=epochs, shuffle=False, validation_data=(X_val, y_val), verbose=True,
                        callbacks=keras_callbacks)

    print(history.history.keys())

    history_dict = history.history
    loss_values = history_dict['loss']
    categorical_crossentropy = history_dict['val_mse']
    epochs = range(1, len(loss_values) + 1)

    plt.plot(epochs, categorical_crossentropy, 'r', label='val_mse')
    plt.plot(epochs, loss_values, 'b', label='loss')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    model.evaluate(X_test, y_test, verbose=True)
    model.save('./my_model_recurrent')
    return model


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
    return keras.models.load_model('my_model_recurrent')


def reset_model(model):
    model.reset_states()


# TODO creare dati per simulare predizione
# TODO scaricare più dati da wyscout

# parse_for_prediction()
# set_gpu_on()
# load_parse_and_merge()
# model = load_model()
# reset_model(model)
# model.evaluate(X_test, y_test)
# model = get_model()
# run_model(model, 1000)
# model.summary()
# sample_predict(model)
