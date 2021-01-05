# %%

import tensorflow as tf
import matplotlib.pyplot as plt
import os

print(tf.__version__)

# %%

BATCH_SIZE = 32
IMG_SIZE = (100, 120)

# %%
CLASS = ('E', 'I')

TARGET_PATH = lambda x: f"data/{CLASS[0]}{CLASS[1]}/{x}"

# %%

os.chdir('C:\\Users\\leeh9\\UGRP-MBTI\\2020-ugrp-Face-analysis-MBTI')
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(TARGET_PATH("train"),
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(TARGET_PATH("validation"),
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  image_size=IMG_SIZE)
# %%
class_names = train_dataset.class_names

# Create test dataset from validation dataset
val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)

# prefetch image for faster I/O during model traning
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

# data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

# Rescal pixel value between [0, 255] to [-1, 1]
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 127.5, offset=-1)

# Create the base model from the pre-trained model Xception
# To use other model simply change Xception to other model name
# check out all available model in https://keras.io/api/applications/
IMG_SHAPE = IMG_SIZE + (3,)

# %%
def plot_accuracy_loss(acc, val_acc, loss, val_loss):
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label= 'Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

# %%
n_input = 20
n_hidden_1 = 20
n_hidden_2 = 20
n_hidden_3 = 20
n_hidden_4 = 20
n_hidden_5 = 20
n_hidden_6 = 20
n_output = 1

initializer = tf.keras.initializers.LecunNormal()
base_model = tf.keras.applications.ResNet50(input_shape=IMG_SHAPE, include_top=True)
base_model.trainable = False

MPL_model = tf.keras.Sequential([
    tf.keras.layers.Dense(input_shape=(128, 1), units=n_hidden_1, activation=tf.nn.selu, name='hidden_1', kernel_initializer=initializer),
    tf.keras.layers.Dense(units=n_hidden_2, activation=tf.nn.selu, name='hidden_2', kernel_initializer=initializer),
    tf.keras.layers.Dense(units=n_hidden_3, activation=tf.nn.selu, name='hidden_3', kernel_initializer=initializer),
    tf.keras.layers.Dense(units=n_hidden_4, activation=tf.nn.selu, name='hidden_4', kernel_initializer=initializer),
    tf.keras.layers.Dense(units=n_hidden_5, activation=tf.nn.selu, name='hidden_5', kernel_initializer=initializer),
    tf.keras.layers.Dense(units=n_hidden_6, activation=tf.nn.selu, name='hidden_6', kernel_initializer=initializer),
    tf.keras.layers.Dense(units=n_output, activation=tf.nn.selu, name='output', kernel_initializer=initializer),
])

inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
outputs = MPL_model(x)
model = tf.keras.Model(inputs, outputs)

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

total_epochs=20
model.fit(train_dataset,
         epochs=total_epochs, # batch size?
         validation_data=validation_dataset)

print("HI")

# Plot training result

acc = model.history['accuracy']
val_acc = model.history['val_accuracy']
loss = model.history['loss']
val_loss = model.history['val_loss']

plot_accuracy_loss(acc, val_acc, loss, val_loss)

# Evaluation of model
loss, accuracy = model.evaluate(test_dataset)
print('Test accuracy :', accuracy)

# Retrieve a batch of images from the test set
image_batch, label_batch = test_dataset.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch).flatten()

# Apply a sigmoid since our model returns logits
predictions = tf.nn.sigmoid(predictions)
predictions = tf.where(predictions < 0.5, 0, 1)

print('Predictions:\n', predictions.numpy())
print('Labels:\n', label_batch)

# %%

# Different number of numbers at each layergit reset --hard HEAD~3
# Batch normalization
