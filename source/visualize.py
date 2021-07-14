import os
import numpy as np
import argparse
from ezconfigparser import Config
from data.loader import get_dataset
from model.model import ExtractorNetwork, ContextNetwork
import tensorflow as tf

# --- hard-coded parameters --- #

TEMPLATE_DATA_CFG = os.path.join(os.path.dirname(__file__), 'cfg', 'gen_data.cfg')
TEMPLATE_MODEL_CFG = os.path.join(os.path.dirname(__file__), 'cfg', 'model.cfg')
VIEW_SIZE = 4
PRED_SIZE = 2
NEG_RATIO = 5

# --------- functions --------- #

def build_cfg(args):
    data_cfg = Config(TEMPLATE_DATA_CFG)
    model_cfg = Config(TEMPLATE_MODEL_CFG)
    data_cfg.parse(args.data_cfg)
    model_cfg.parse(args.model_cfg)
    data_cfg.merge(model_cfg)
    return data_cfg

def calc_feat_length(cfg):
    length = (160 - cfg.kernel_size) // 2 + 1
    length = (length - cfg.kernel_size) // 2 + 1
    return length

def calc_feat_size(cfg):
    dim = (60 - cfg.kernel_size) // 2 + 1
    dim = (dim - cfg.kernel_size) // 2 + 1
    dim *= cfg.filter_size
    return dim

@tf.function
def train_step(img, label, label_length):
    with tf.GradientTape() as tape:
        feat = extractor_net(img, training=True)
        context = tf.signal.frame(feat, VIEW_SIZE, 1, axis=1) # [B, T, VIEW_SIZE, D]
        context = tf.reshape(context, [context.shape[0], context.shape[1], -1]) # [B, T, VIEW_SIZE * D]
        pred = context_net(context[:, :-1, :])

        # Get Positive Latent Feature
        pos_context = []
        for indx in range(PRED_SIZE):
            pos_context.append(feat[:, VIEW_SIZE + 1 + indx: feat_len - PRED_SIZE + indx + 1, :])
        
        loss = 0.
        for indx in range(PRED_SIZE):
            loss -= tf.math.log(tf.nn.sigmoid(tf.reduce_sum(tf.math.multiply(pos_context[indx], pred[indx]), axis=2)) + 1e-6)

        # Get Negative Latent Feature
        batch_shift = tf.random.uniform([NEG_RATIO], 1, img.shape[0] - 1, dtype=tf.int32)
        time_shift = tf.random.uniform([NEG_RATIO], 1, feat_len - PRED_SIZE - VIEW_SIZE - 1, dtype=tf.int32)
        
        for indx in range(PRED_SIZE):
            shifted = tf.roll(pos_context[indx], batch_shift[indx], axis=0)
            shifted = tf.roll(shifted, time_shift[indx], axis=1)
            loss -= 1 / PRED_SIZE * tf.math.log(tf.nn.sigmoid(- tf.reduce_sum(tf.math.multiply(shifted, pred[indx]), axis=2)) + 1e-6)
        
        loss = tf.reduce_mean(loss, axis=1)

    gradients = tape.gradient(loss, extractor_net.trainable_variables + context_net.trainable_variables)
    optimizer.apply_gradients(zip(gradients, extractor_net.trainable_variables + context_net.trainable_variables))
    train_loss(loss)


@tf.function
def valid_step(img, label, label_length):
    
    feat = extractor_net(img, training=False)
    context = tf.signal.frame(feat, VIEW_SIZE, 1, axis=1) # [B, T, VIEW_SIZE, D]
    context = tf.reshape(context, [context.shape[0], context.shape[1], -1]) # [B, T, VIEW_SIZE * D]
    
    return feat


@tf.function
def decode_step(img, label, label_length):
    logit = model(img, training=False)
    logit = tf.transpose(logit, perm=[1, 0, 2])
    decoded, log_probability = tf.nn.ctc_beam_search_decoder(logit, label_length, 10, 1)
    return decoded, log_probability

# ----- main starts here ------ #

parser = argparse.ArgumentParser(description='CTC Train')
parser.add_argument('-d', '--data_cfg', type=str, help='data configuration', required=True)
parser.add_argument('-m', '--model_cfg', type=str, help='model configuration', required=True)
args = parser.parse_args()
cfg = build_cfg(args)

feat_size = calc_feat_size(cfg)
feat_len = calc_feat_length(cfg)

extractor_net = ExtractorNetwork(cfg)
extractor_net.load_weights(cfg.model_dir + '.extract')
# context_net = ContextNetwork(cfg, PRED_SIZE, feat_size)
# context_net.load_weights(cfg.model_dir + '.context')

train_ds, valid_ds = get_dataset(cfg, mode='train')



for data in valid_ds:
    pos_feat = valid_step(data[0], data[1], data[2]).numpy()[0]
    mat = np.zeros((pos_feat.shape[0], pos_feat.shape[0]))
    for i in range(pos_feat.shape[0]):
        for j in range(pos_feat.shape[0]):
            mat[i][j] = np.sum(pos_feat[i] * pos_feat[j]) / np.linalg.norm(pos_feat[i]) / np.linalg.norm(pos_feat[j])
    import matplotlib.pyplot as plt
    plt.matshow(mat)
    plt.show()

    plt.imshow(data[0][0].numpy())
    plt.show()
    exit(0)

