import numpy as np
import time


def generate_1d_representation(num_entities, num_timestamps, num_symbols, entities_idx, symbols_idx, data_df):
    start_time = time.time()
    
    rep1d = np.zeros((num_entities, num_timestamps, num_symbols))
    
    # fill representation matrix
    for idx, sti in data_df.iterrows():
        rep1d[entities_idx[sti['entity']], sti['start']: sti['finish'] + 1, symbols_idx[sti['symbol']]] += 1    
        
    return rep1d, time.time() - start_time


def generate_representation(labels_df, data_df, pad_to_block_size=False, block_size=None):
    # get indices of entities, labels and symbols
    entities_idx = {v: k for k, v in enumerate(labels_df['entity'].unique())}
    labels_idx = {v: k for k, v in enumerate(labels_df['label'].unique())}
    symbols_idx = {v: k for k, v in enumerate(data_df['symbol'].unique())}
    
    # create mapping from entity index to label index
    entities_labels = np.zeros((labels_df['entity'].nunique()))
    for idx, entity in labels_df[['entity', 'label']].drop_duplicates().iterrows():
        entities_labels[entities_idx[entity['entity']]] = labels_idx[entity['label']]
    
    # generate representation matrix 
    num_entities, num_symbols, num_timestamps = len(entities_idx), len(symbols_idx), data_df['finish'].max() + 1
    if pad_to_block_size and block_size is not None and num_timestamps > block_size:
        num_timestamps = block_size * (num_timestamps // block_size + 1)
    rep1d, generation_time = generate_1d_representation(
        num_entities, num_timestamps, num_symbols,
        entities_idx, symbols_idx, data_df
    )
        
    return rep1d, entities_labels, generation_time
