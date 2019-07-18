from __future__ import absolute_import, division, print_function, unicode_literals


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import random

random.seed(42)


iris = load_iris()


X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, random_state=1)


one_hot = OneHotEncoder()


# X_train = X_train.astype('int8')
# y_train = y_train.astype('float32')

y_train = one_hot.fit_transform(y_train.reshape(-1, 1))
y_test = one_hot.transform(y_test.reshape(-1, 1))




model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Dense(10, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(3, activation='softmax'),

    ]
)


opt = tf.keras.optimizers.SGD(learning_rate=0.1)


model.compile(optimizer=opt,
              loss='categorical_crossentropy' ,
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10)