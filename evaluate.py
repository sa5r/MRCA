import datetime
import time
import tensorflow as tf
from typing import Dict, List
import argparse
from utils import load_glove_dictionary, encode_relations, process_data
from model import RC_DiceLoss, MRCA

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required = True, )
    parser.add_argument('--relations_path', required = True, )
    parser.add_argument('--padding_size', required = True, type = int)
    parser.add_argument('--glove_path', required = True, )
    parser.add_argument('--embedding_dimensions', required = True, type = int)
    parser.add_argument('--lstm_units', required = True, type = int)
    parser.add_argument('--pool_size', required = True, type = int)
    parser.add_argument('--strides', required = True, type = int)
    parser.add_argument('--dropout', required = True, type = float)
    parser.add_argument('--batch_size', required = True, type = int)
    parser.add_argument('--checkpoint_path', required = True, )
    args = parser.parse_args()

    # load Glove
    glove_dictionary, vector_value = load_glove_dictionary(args.glove_path)

    # load relations set
    relations_encoded = encode_relations(args.relations_path)

    # process training input
    test_X, test_Y      = process_data(path = args.data_path,
        relations = relations_encoded,
        vectors = glove_dictionary,
        vector_value = vector_value,
        padding_size = args.padding_size,
        embedding_dimensions = args.embedding_dimensions)

    # create the model
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
        # learning_rate = args.learning_rate,
    ),
    loss = RC_DiceLoss(),
    metrics = (
        tf.keras.metrics.Precision(name = 'precision'),
        tf.keras.metrics.Recall(name = 'recall')
    ),
    )

    # load trained model
    model.load_weights(args.checkpoint_path)

    # perform prediction
    predictions = model.predict(
        x =           test_X['embeddings'],
        batch_size =  args.batch_size,
    )

    ground_truth_count = predicted_count = correct_count = 0

    # iterate over predictions
    for i, ground_truth in enumerate(test_Y['relations_text']):

        ground_truth_rel_list = ground_truth.split('^sep^')
        ground_truth_count += len(ground_truth_rel_list)

        for j, pred_probability in enumerate(predictions[i]):     

            if pred_probability >= 0.5:
                predicted_count += 1
                rel = relations_encoded[j]
                if rel in ground_truth_rel_list:
                    correct_count += 1
  
    precision = correct_count / predicted_count
    recall = correct_count / ground_truth_count
    f1_score = 2 * precision * recall / (precision + recall)

    print('\n')
    print(f'correct_count:{correct_count}, predicted_count:{predicted_count}, ground_truth_count:{ground_truth_count}')
    print(f'precision :{precision}, recall :{recall}, f1 :{f1_score}')
    print('\n')

if __name__ == "__main__":
    main()