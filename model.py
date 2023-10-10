import tensorflow as tf
import tensorflow.keras.layers as layers

class RC_DiceLoss(tf.keras.losses.Loss):
    def __init__(self, smooth = 1e-6, gama = 2):
        super(RC_DiceLoss, self).__init__()
        self.name = 'RCDL'
        self.smooth = smooth
        self.gama = gama

    def call(self, y_true, y_pred):
        y_true, y_pred = tf.cast(y_true, dtype=tf.float32), tf.cast(y_pred, tf.float32)
        return tf.where(
            tf.logical_and(
                tf.equal(y_true, tf.constant(0.0)),
                tf.less(y_pred, tf.constant(0.5)),
            ),
            tf.divide(
                # nominator
                self.smooth ** 2,
                # denominator
                tf.reduce_sum(y_pred ** self.gama) + tf.reduce_sum(y_true ** self.gama) + self.smooth
            ),
            1 - tf.divide(
                # nominator
                2 * tf.reduce_sum(tf.multiply(y_pred, y_true)) + self.smooth,
                # denominator
                tf.reduce_sum(y_pred ** self.gama) + tf.reduce_sum(y_true ** self.gama) + self.smooth
            )
        )

class MRCA(tf.keras.Model):
    def __init__(self,
        padding_size: int,
        embedding_dimensions: int,
        relations_size: int,
        lstm_units: int,
        pool_size: int,
        strides: int,
        dropout: float,
        ):

        super().__init__()

        # input layer with input shape
        # add 1 for the case vector and 1 for the entity vector
        self.inputs = layers.Input(shape = (padding_size, embedding_dimensions + 1 + 1 ))

        # bi-lstm with tanh
        self.bilstm = layers.Bidirectional(
            layers.LSTM(lstm_units , return_sequences = True, activation = 'tanh',)
        )

        # average pooling layer to reduce dimensionality
        self.avg = layers.AveragePooling1D(pool_size = pool_size, strides = strides, padding = 'same')

        # flatten 2D to 1D
        self.flt = layers.Flatten()

        # overfitting
        self.dropout_layer = layers.Dropout(dropout)

        # output layer that has units number equal to relations size
        # with no activation function
        self.output_layer = layers.Dense(relations_size,
            name = 'output_layer',
            activation = None
        )

    def call(self, inputs):
        x = self.bilstm(inputs)
        x = self.avg(x)
        x = self.flt(x)
        x = self.dropout_layer(x)
        return self.output_layer(x)