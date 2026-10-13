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
from PredictionDataBuilder import PredictionDataBuilder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from Merger import Merger
from Parser import Parser

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


def load_parse_and_merge():
    files = []

    for file in listdir('./data'):
        file = './data/' + file
        files.append(file)

    total = len(files)

    for idx, file in enumerate(files):
        ps = Parser()
        print('Parsing file number ' + str(idx + 1) + ' with name ' + file.title() + ' of ' + str(total))
        ps.parse_excel(file)
        break

    files = []

    for file in listdir('./second-step-data'):
        file = './second-step-data/' + file
        files.append(file)

    nm = Merger()
    nm.load_excels(files)


# Define the prior weight distribution as Normal of mean=0 and stddev=1.
# Note that, in this example, the we prior distribution is not trainable,
# as we fix its parameters.


keras_callbacks = [tensorboard, EarlyStopping()]

games = pd.read_excel("./out/out.xlsx")

games.drop('Score-Differential', axis=1, inplace=True)
games.sort_index(ascending=False)

X = pd.DataFrame(columns=games.columns)
y = pd.DataFrame(columns=games.columns)


def split_input_outputs(data_X: pd.DataFrame, train_x: pd.DataFrame, train_y: pd.DataFrame):
    stopper = 0
    X = train_x
    y = train_y
    for idx, row in data_X.iterrows():
        if idx == 4 - 1 + stopper:
            stopper += 4
            df_y = row.to_frame().T
            temp_y = pd.concat([df_y, y]).reset_index(drop=True)
            y = temp_y
        else:
            df_x = row.to_frame().T
            temp_x = pd.concat([df_x, X]).reset_index(drop=True)
            X = temp_x
        print('working on row number ' + str(idx))
    return X, y


X, y = split_input_outputs(games, X, y)

scaler_X = QuantileTransformer().fit(X)
scaler_y = QuantileTransformer().fit(y)


X_train = scaled_X = pd.DataFrame(scaler_X.transform(X), index=X.index, columns=X.columns)
y_train = scaled_y = pd.DataFrame(scaler_y.transform(y), index=y.index, columns=y.columns)

print(X_train, y_train)

X_train, y_train = np.array(scaled_X), np.array(scaled_y)

X_train = np.reshape(X_train, (y_train.shape[0], 3, X_train.shape[1]))
y_train = np.reshape(y_train, (y_train.shape[0], 1, y_train.shape[1]))

print("X_TRAIN SHAPE")
print(X_train.shape)
print("Y_TRAIN SHAPE")
print(y_train.shape)

X_train, X_test = train_test_split(X_train, test_size=0.33, random_state=False)
y_train, y_test = train_test_split(y_train, test_size=0.33, random_state=False)

X_train, X_val = train_test_split(X_train, test_size=0.25, random_state=False)
y_train, y_val = train_test_split(y_train, test_size=0.25, random_state=False)

model_input_shape = (X_train.shape[0], X_train.shape[1])
output_shape = y_train.shape[1]


def get_model():
    activation = tf.keras.activations.tanh
    recurrent_activation = tf.keras.activations.sigmoid
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
            LSTM(units=3,
                 input_shape=(208, 3),
                 activation=activation,
                 recurrent_activation=recurrent_activation,
                 return_sequences=True,
                 name='Bidirectional')
        )
    )
    # model.add(drop_out_layer)
    for i in range(7, 12):
        model.add(LSTM(units=3, return_sequences=True, name='LSTM_' + str(i)))

        model.add(drop_out_layer)
    model.add(Dense(output_shape, activation=activation, name='Dense_Output'))

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
    model.save('./my_model/saved_model.h5')
    return model


def sample_predict(best_model, all_games: pd.DataFrame):
    PredictionDataBuilder('Leicester City')

    data = pd.read_excel('./temp_prediction/out.xlsx')

    scaled_predict = scaler_X.transform(data)

    scaled_predict = np.array(scaled_predict)

    raw_predictions = best_model.predict(scaled_predict)
    predictions_max = np.argmax(raw_predictions, axis=1)
    prediction = scaler_y.inverse_transform(predictions_max)

    print(raw_predictions)
    print(prediction)


def load_model():
    return keras.models.load_model('./my_model/saved_model.h5')


def reset_model(model):
    model.reset_states()


set_gpu_on()
# load_parse_and_merge()
# model = load_model()
# reset_model(model)
# model.evaluate(X_test, y_test)
model = get_model()
# run_model(model, 2)
# model.summary()
sample_predict(model, games)
