import os
import argparse
from ezconfigparser import Config
from data.loader import get_dataset
from model.model import CTCClassifier
import tensorflow as tf

# --- hard-coded parameters --- #

TEMPLATE_DATA_CFG = os.path.join(os.path.dirname(__file__), 'cfg', 'gen_data.cfg')
TEMPLATE_MODEL_CFG = os.path.join(os.path.dirname(__file__), 'cfg', 'model.cfg')

# --------- functions --------- #

def build_cfg(args):
    data_cfg = Config(TEMPLATE_DATA_CFG)
    model_cfg = Config(TEMPLATE_MODEL_CFG)
    data_cfg.parse(args.data_cfg)
    model_cfg.parse(args.model_cfg)
    data_cfg.merge(model_cfg)
    return data_cfg

def calc_logit_length(cfg):
    length = (160 - cfg.kernel_size) // 2 + 1
    length = (length - cfg.kernel_size) // 2 + 1
    length = (length - cfg.kernel_size) // 2 + 1
    return length

@tf.function
def train_step(img, label, label_length):
    with tf.GradientTape() as tape:
        logit = model(img, training=True)
        _logit_length = tf.ones([img.shape[0]], dtype=tf.int32) * logit_len
        loss = loss_fn(label, logit, label_length, _logit_length, logits_time_major=False)
        loss = tf.reduce_mean(loss)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)


@tf.function
def valid_step(img, label, label_length):
    logit = model(img, training=False)
    _logit_length = tf.ones([img.shape[0]], dtype=tf.int32) * logit_len
    loss = loss_fn(label, logit, label_length, _logit_length, logits_time_major=False)
    valid_loss(loss)


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

model = CTCClassifier(cfg)

train_ds, valid_ds = get_dataset(cfg, mode='train')

optimizer = tf.keras.optimizers.RMSprop(cfg.learning_rate)
loss_fn = tf.nn.ctc_loss

valid_loss = tf.keras.metrics.Mean(name='valid_loss')
train_loss = tf.keras.metrics.Mean(name='train_loss')

logit_len = calc_logit_length(cfg)

total_step = '?'
best_valid_loss = 1e10
best_valid_epoch = -1

for epoch in range(cfg.epoch):
    step = 0
    train_loss.reset_states()
    valid_loss.reset_states()

    for data in train_ds:
        train_step(data[0], data[1], data[2])
        print(f'Epoch {epoch + 1}, '
            f'Step {step} / {total_step}, '
            f'Loss: {train_loss.result():.6f}', end='\r')
        step += 1
    print(f'Epoch {epoch + 1}, '
        f'Step {step} / {total_step}, '
        f'Loss: {train_loss.result():.6f}')
    
    for data in valid_ds:
        valid_step(data[0], data[1], data[2])
    print(f'Epoch {epoch + 1}, Validation Loss: {valid_loss.result():.6f}')

    if valid_loss.result() < best_valid_loss:
        best_valid_loss = valid_loss.result()
        best_valid_epoch = epoch + 1
        model.save_weights(cfg.model_dir)
    elif (epoch + 1) - best_valid_epoch > cfg.patience:
        break

    total_step = step

print(f'Saved Weights on Epoch {best_valid_epoch}')