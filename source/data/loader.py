import os
import numpy as np
import random
import tensorflow as tf

# --- hard-coded parameters --- #
random.seed(123)

# --------- functions --------- #

def get_dataset(cfg, mode='train'):
    
    def _parse_fn(img_dir, indexes, lengths):
        img = tf.dtypes.cast(tf.image.decode_image(tf.io.read_file(img_dir)), dtype=tf.float32)
        img = (tf.transpose(img, perm=[1, 0, 2]) - 127.) / 255.
        return (img, indexes, lengths)

    with open(os.path.join(cfg.data_dir, 'data.txt'), 'r') as f:
        _ = int(f.readline())
        content = [x.strip('\n').split('\t') for x in f.readlines() if len(x.strip('\n')) > 2]
    random.shuffle(content)
    lengths = [int(x[0]) for x in content]
    lengths = np.array(lengths, dtype=np.int32)
    indexes = [[int(y) for y in x[1:-1]] for x in content]
    indexes = np.array(indexes, dtype=np.int32)
    imgs = [x[-1] for x in content]
    imgs = np.array(imgs)

    if mode == 'train':
        split_index = round(len(content) * cfg.train_split)
        train_lengths = lengths[:split_index]
        valid_lengths = lengths[split_index:]
        train_indexes = indexes[:split_index]
        valid_indexes = indexes[split_index:]
        train_imgs = imgs[:split_index]
        valid_imgs = imgs[split_index:]

        train_ds = tf.data.Dataset.from_tensor_slices((train_imgs, train_indexes, train_lengths)) \
            .shuffle(50000) \
            .map(map_func=_parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
            .batch(cfg.batch_size)
        
        valid_ds = tf.data.Dataset.from_tensor_slices((valid_imgs, valid_indexes, valid_lengths)) \
            .map(map_func=_parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
            .batch(cfg.batch_size)

        return train_ds, valid_ds

    else:
        test_ds = tf.data.Dataset.from_tensor_slices((imgs, indexes, lengths)) \
            .map(map_func=_parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
            .batch(cfg.batch_size)
        return test_ds

    