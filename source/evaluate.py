import os
import argparse
from ezconfigparser import Config
from data.loader import get_dataset
from model.model import CTCClassifier
import tensorflow as tf
from jiwer import wer

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
    # length = (length - cfg.kernel_size) // 2 + 1
    return length


@tf.function
def decode_step(img):
    logit = model(img, training=False)
    logit = tf.transpose(logit, perm=[1, 0, 2])
    logit = tf.roll(logit, -1, 2)
    _logit_length = tf.ones([img.shape[0]], dtype=tf.int32) * logit_len
    decoded, log_probability = tf.nn.ctc_beam_search_decoder(logit, _logit_length, 200, 1)
    return decoded, log_probability


def calc_wer(label, pred):
    _label = ' '.join([str(x) for x in label if x > 0])
    _pred = ' '.join([str(x + 1) for x in pred if x >= 0])
    word_error_rate = wer(_label, _pred)
    return word_error_rate

# ----- main starts here ------ #

parser = argparse.ArgumentParser(description='CTC Train')
parser.add_argument('-d', '--data_cfg', type=str, help='test data configuration', required=True)
parser.add_argument('-m', '--model_cfg', type=str, help='model configuration', required=True)
args = parser.parse_args()
cfg = build_cfg(args)

model = CTCClassifier(cfg)
model.load_weights(cfg.model_dir)

test_ds = get_dataset(cfg, mode='test')

total = 0.
total_wer = 0.
total_exact = 0.

logit_len = calc_logit_length(cfg)

for data in test_ds:
    decoded, log_prob = decode_step(data[0])
    decoded = tf.sparse.to_dense(decoded[0], default_value=-1).numpy().tolist()
    label = data[1].numpy().tolist()
    
    for i in range(len(label)):
        total += 1
        word_error_rate = calc_wer(label[i], decoded[i])
        total_wer += word_error_rate
        total_exact += (word_error_rate == 0)


print(f'Number of samples in test set: {int(total):d}')
print(f'Average WER on test set: {total_wer / total:.4f}')
print(f'Exact ACC on test set: {total_exact / total:.4f}')
