import os
import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns
import tensorflow as tf
import pathlib

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models

from AudioRecognition import get_command_data, check_data_set, shuffle_audiofile_after_extract, division_train_val_test
from AudioRecognition import preprocess_dataset

AUTOTUNE = tf.data.AUTOTUNE
data_dir = pathlib.Path("data/mini_speech_commands")

def audio_cnn_model(commands, spectrogram_ds):
    for spectrogram, _ in spectrogram_ds.take(1):
        input_shape = spectrogram.shape
    print('Input shape :', input_shape)
    num_labels = len(commands)
    print('num_labels :', num_labels)

    norm_layer = preprocessing.Normalization()
    norm_layer.adapt(spectrogram_ds.map(lambda x, _: x))

    model = models.Sequential([
        layers.Input(shape=input_shape),
        preprocessing.Resizing(32, 32),
        norm_layer,
        layers.Conv2D(32, 3, activation='relu'),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_labels),
    ])
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )
    return model


def train_model(model, train_ds, val_ds):
    EPOCHS = 10
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
    )
    metrics = history.history
    plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
    plt.legend(['loss'], 'val_loss')
    plt.show()
    return model


def show_confusion_matrix_testset(commands, y_true, y_pred):
    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    #sns.heatmap(confusion_mtx, xticklabels=commands, yticklabels=commands, annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.show()


def evaluate_performance_testset(model, test_ds):
    test_audio = []
    test_labels = []

    for audio, label in test_ds:
        test_audio.append(audio.numpy())
        test_labels.append(label.numpy())

    test_audio = np.array(test_audio)
    test_labels = np.array(test_labels)

    y_pred = np.argmax(model.predict(test_audio), axis=1)
    y_true = test_labels
    test_acc = sum(y_pred == y_true) / len(y_true)
    print(f'Test set accuracy : {test_acc : .0%}')


def inference_from_audio_file(model, commands):
    sample_file = data_dir/'no/01bb6a2a_nohash_0.wav'
    sample_ds = preprocess_dataset([str(sample_file)])

    for spectrogram, label in sample_ds.batch(1):
        prediction = model(spectrogram)
        plt.bar(commands, tf.nn.softmax(prediction[0]))
        plt.title(f'Prediction for "{commands[label[0]]}"')
        plt.show()


def convert_audio_model_tflite(model):
    #Convert the model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open("converted_audio_model.tflite", "wb").write(tflite_model)


def main():
   get_command_data()
   commands = check_data_set()
   filenames = shuffle_audiofile_after_extract(commands)
   train_files, val_files, test_files = division_train_val_test(filenames)

   train_ds = preprocess_dataset(train_files)
   val_ds = preprocess_dataset(val_files)
   test_ds = preprocess_dataset(test_files)

   train_ds = train_ds.cache().prefetch(AUTOTUNE)
   val_ds = val_ds.cache().prefetch(AUTOTUNE)
   model = audio_cnn_model(commands, train_ds)

   batch_size = 64
   train_ds = train_ds.batch(batch_size)
   val_ds = val_ds.batch(batch_size)
   trained_model = train_model(model, train_ds, val_ds)
   #evaluate_performance_testset(trained_model, test_ds)
   #inference_from_audio_file(trained_model, commands)
   convert_audio_model_tflite(trained_model)

if __name__ == "__main__":
    main()
