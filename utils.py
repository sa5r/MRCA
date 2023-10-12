from typing import Dict, List
import numpy as np
from tqdm import tqdm
import json
import sys
import re

def load_glove_dictionary(path: str, ) -> Dict:

    max_vector = 0
    vectors = {}
    with open(path, 'r') as f:
        for i, line in enumerate(tqdm(f)):

            # Glove has values separated with space
            vals = line.rstrip().split(' ')
            embeddings = [float(x) for x in vals[1:]]

            # To find the maximum vector value in dictionary
            mx = max(embeddings)
            if max_vector < mx:
                max_vector = mx

            vectors[vals[0]] = embeddings

    max_vector = max_vector + 1
    print(f'\nMax value found \t {max_vector}\n')
    print(f'\nVocab size \t {len( vectors.keys() )}\n')
    return vectors, max_vector

def encode_relations(path: str, ) -> List:
    with open(path, 'r') as f:
        rel_json = json.load(f)
        relations = [item for item in rel_json]
        print(f'\nRelations count \t {len(relations)}\n')
        return relations
    
def process_data(path: str, relations: List, vectors: Dict, vector_value: float, padding_size: int, embedding_dimensions: int, size = 1) -> np.ndarray:

    def get_word_embedding(word: str, ) -> List:

        # First char case vector
        case_vector = vector_value if word[0].isupper() else (-1 * vector_value)

        wrd = str(word).lower()
        
        # Check if found
        if wrd in vectors:
            return vectors[wrd] + [case_vector]
        
        # Else
        char_embeddings = [
            vectors.setdefault(c, vectors[dict_keys[ord(c) % 100]]) for c in wrd
        ]
        averaged_embeddings = []
        tot = 0
        for i in range(embedding_dimensions):
            for j in range(len(char_embeddings)):
                tot += char_embeddings[j][i]
            averaged_embeddings.append(tot)

        return averaged_embeddings + [case_vector]
    
    # Loading vector keys
    dict_keys = list(vectors.keys())

    # load file
    print(f'open {path}')
    f = open(path, 'r')
    corpus = json.load(f)
    f.close()

    X = {
        'sentences' : [], # raw text
        'embeddings' : [], # for relations task
    }
    Y = {
        'relations' : [], # binary encoded ground truth
        'relations_text' : [], # raw ground truth
    }

    # iterate over input items
    for i in tqdm( range( int( len(corpus) * size ) ) ):
        sentence_data = corpus[i]
        tokens = sentence_data['text'].split(' ')

        # store raw text
        X['sentences'].append( sentence_data['text'] )

        # initial variables
        embeddings = []
        relations_tagged = [0] * len(relations)
        relations_text = []
        sub_ranges = []
        obj_ranges = []

        # triples iteration
        for trpl in sentence_data['relation_list']:
            rel = trpl['predicate']
            relations_text.append(rel)

            # mandatory to have the relation pre-defined
            if rel not in relations:
                exit(f'relation {rel} not found.')

            rel_index = relations.index(rel)

            # assign ground truth with positive label
            relations_tagged[rel_index] = 1

            # subjects set ranges
            sub_list = trpl['subject'].split(' ')
            sub_head_index = tokens.index(sub_list[0])
            sub_tail_index = tokens.index(sub_list[-1], sub_head_index)
            assert sub_tail_index != -1
            sub_ranges.append([sub_head_index, sub_tail_index])

            # objects set ranges
            obj_list = trpl['object'].split(' ')
            obj_head_index = tokens.index(obj_list[0])
            obj_tail_index = tokens.index(obj_list[-1], obj_head_index)
            assert obj_tail_index != -1
            obj_ranges.append([obj_head_index, obj_tail_index])
        
        Y['relations'].append(relations_tagged)

        # remove duplicates
        relations_text = list(set(relations_text))
        Y['relations_text'].append('^sep^'.join(relations_text))

        # encoder part for added entity state vector
        # iterate over words and add the vactor
        for j, tkn in enumerate(tokens):
            is_sub = is_obj = False

            # check if the word is a subject, object or none
            for s_rng in sub_ranges:
                if s_rng[0] <= j <= s_rng[1]:
                    is_sub = True
                    break
            for o_rng in obj_ranges:
                if o_rng[0] <= j <= o_rng[1]:
                    is_obj = True
                    break

            if is_sub: embeddings.append( get_word_embedding(tkn) + [vector_value] )
            elif is_obj: embeddings.append( get_word_embedding(tkn) + [-1 * vector_value] )
            else: embeddings.append( get_word_embedding(tkn) + [0] )
        
        # sequence padding
        padding_filling = (padding_size - len(embeddings)) * [(embedding_dimensions + 1 + 1 ) * [0.0]]
        embeddings.extend(padding_filling)

        # add ready embeddings
        X['embeddings'].append(embeddings)

    # convert to numpy to reduce memory usage
    for k in X.keys():
        X[k] = np.array(X[k])
        print( f'{k}\tshape\t{ X[k].shape }\tsize\t{sys.getsizeof(X[k]) / 1024 / 1024:.4} MB',  )

    for k in Y.keys():
        Y[k] = np.array(Y[k])
        print( f'{k}\tshape\t{ Y[k].shape }\tsize\t{sys.getsizeof(Y[k]) / 1024 / 1024:.4} MB',  )

    return X, Y
