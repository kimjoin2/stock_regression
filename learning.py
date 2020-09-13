import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from datetime import datetime


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        print(datetime.now(), epoch, logs)


column_names = ['high', 'low', 'open', 'close', 'volume', 'price_up_rate']


def learning():
    raw_dataset = pd.read_csv("preprocessed_data/train_data.csv",
                              names=column_names,
                              skipinitialspace=True)
    dataset = raw_dataset.copy()
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    train_labels = train_dataset.pop('price_up_rate')
    test_labels = test_dataset.pop('price_up_rate')

    model = keras.Sequential([
        layers.Dense(5, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(120, activation='relu'),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])

    history = model.fit(
        train_dataset, train_labels,
        epochs=10, validation_split=0.3, verbose=1,
        batch_size=len(train_dataset),
        callbacks=[PrintDot()])

    loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=1)

    print('test : ', loss, mae, mse)

    plot_history(history)


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure(figsize=(8, 12))

    plt.subplot(2, 1, 1)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error')
    plt.plot(hist['epoch'], hist['mae'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'],
             label='Val Error')
    plt.ylim([0, 15])
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mse'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'],
             label='Val Error')
    plt.ylim([0, 15])
    plt.legend()
    plt.show()


def show_relation():
    raw_dataset = pd.read_csv("preprocessed_data/train_data.csv",
                              names=column_names,
                              skipinitialspace=True)
    train_dataset = raw_dataset.sample(frac=0.005, random_state=0)
    sns.pairplot(train_dataset[['high', 'low', 'open', 'close', 'volume', 'price_up_rate']], diag_kind="kde")
    plt.show()


learning()
# show_relation()
print('done')
