import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        print(epoch, logs)


column_names = ['high', 'low', 'open', 'close', 'volume', 'price_up']


def learning():
    raw_dataset = pd.read_csv("preprocessed_data/train_data.csv",
                              names=column_names,
                              skipinitialspace=True)
    dataset = raw_dataset.copy()
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    train_labels = train_dataset.pop('price_up')
    test_labels = test_dataset.pop('price_up')

    model = keras.Sequential([
        layers.Dense(12, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(6, activation='relu'),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])

    history = model.fit(
        train_dataset, train_labels,
        epochs=50, validation_split=0.2, verbose=0,
        callbacks=[PrintDot()])

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
    plt.ylim([0, 5])
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mse'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'],
             label='Val Error')
    plt.ylim([0, 20])
    plt.legend()
    plt.show()


def show_relation():
    raw_dataset = pd.read_csv("preprocessed_data/train_data.csv",
                              names=column_names,
                              skipinitialspace=True)
    train_dataset = raw_dataset.sample(frac=0.8, random_state=0)
    sns.pairplot(train_dataset[['close', 'volume']], diag_kind="kde")
    plt.show()


learning()
# show_relation()
print('done')
