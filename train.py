import datetime
import time
import random
import tensorflow as tf
from typing import Dict, List
import argparse
from utils import load_glove_dictionary, encode_relations, process_data
from model import RC_DiceLoss, MRCA

class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, patience: int, decay: float):
        self.best_f1 = 0.0
        self.best_weights = None
        self.noprogress_counter = 0
        self.patience = patience
        self.decay = decay
    
    def on_epoch_end(self, epoch, logs = None):

        # Get and set the current learning rate from model's optimizer.
        lr = float( tf.keras.backend.get_value(self.model.optimizer.learning_rate) )
        lr = lr - self.decay
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        print(f'\nLearning rate now is \t %6.6f.' % (lr))

        # Calculate validation F1
        f1 = (2 * logs['val_precision'] * logs['val_recall']) / (logs['val_precision'] + logs['val_recall'] + 0.000001)

        # Get best weights
        if f1 > self.best_f1:
            self.best_f1 = f1
            print(f'\nValidation f1 \t {f1:.4} epoch \t {epoch}')
            self.best_weights = self.model.get_weights()
            self.noprogress_counter = 0
        
        # stop training
        elif self.noprogress_counter > self.patience:
            self.model.set_weights(self.best_weights)
            self.model.stop_training = True

        else:
            self.noprogress_counter += 1
        
    def on_train_end(self, logs = None):
        self.model.set_weights(self.best_weights)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required = True, )
    parser.add_argument('--validation_path', required = True, )
    parser.add_argument('--relations_path', required = True, )
    parser.add_argument('--train_size', required = True, type = float)
    parser.add_argument('--learning_rate', required = True, type = float)
    parser.add_argument('--padding_size', required = True, type = int)
    parser.add_argument('--glove_path', required = True, )
    parser.add_argument('--embedding_dimensions', required = True, type = int)
    parser.add_argument('--lstm_units', required = True, type = int)
    parser.add_argument('--pool_size', required = True, type = int)
    parser.add_argument('--strides', required = True, type = int)
    parser.add_argument('--dropout', required = True, type = float)
    parser.add_argument('--learning_decay', required = True, type = float)
    parser.add_argument('--batch_size', required = True, type = int)
    parser.add_argument('--epochs', required = True, type = int)
    parser.add_argument('--patience', required = True, type = int)
    parser.add_argument('--checkpoint_path', required = True, )
    args = parser.parse_args()

    # load Glove
    glove_dictionary, vector_value = load_glove_dictionary(args.glove_path)

    # load relations set
    relations_encoded = encode_relations(args.relations_path)

    # process training input
    train_X, train_Y = process_data(path = args.data_path,
        size = args.train_size,
        relations = relations_encoded,
        vectors = glove_dictionary,
        vector_value = vector_value,
        padding_size = args.padding_size,
        embedding_dimensions = args.embedding_dimensions
    )

    # process validation input
    valid_X, valid_Y = process_data(path = args.validation_path,
        relations = relations_encoded,
        vectors = glove_dictionary,
        vector_value = vector_value,
        padding_size = args.padding_size,
        embedding_dimensions = args.embedding_dimensions
    )

    # preprocessing ends here

    # create the model
    tf.random.set_seed(random.randint(100, 9999))

    model = MRCA(padding_size = args.padding_size,
        embedding_dimensions = args.embedding_dimensions,
        relations_size = len(relations_encoded),
        lstm_units = args.lstm_units,
        pool_size = args.pool_size,
        strides = args.strides,
        dropout = args.dropout,
    )
    model.compile(
        optimizer = tf.optimizers.Adam(
            learning_rate = args.learning_rate,
        ),
        loss = RC_DiceLoss(),
        metrics = (
            tf.keras.metrics.Precision(name = 'precision'),
            tf.keras.metrics.Recall(name = 'recall')
        ),
    )

    # early stopping
    relationsCallback = CustomCallback(args.patience, args.learning_decay)

    # train
    model.fit(
        x = train_X['embeddings'],
        y = train_Y['relations'],
        batch_size = args.batch_size,
        epochs = args.epochs,
        validation_data = (
            valid_X['embeddings'],
            valid_Y['relations'],
        ),
        verbose = 1,
        callbacks = relationsCallback,
    )

    # save checkpoint
    model.save_weights(args.checkpoint_path, save_format="tf")

if __name__ == "__main__": main()