import os
import random
import subprocess
from ezconfigparser import Config
from captcha.image import ImageCaptcha

# --- hard-coded parameters --- #

TEMPLATE_CFG = os.path.join(os.path.dirname(__file__), 'cfg', 'gen_data.cfg')
ALL_TOKENS = [str(i) for i in range(10)] # + [chr(65 + i) for i in range(26)] + [chr(97 + i) for i in range(26)]

# --------- functions --------- #

def gen_random_seq(cfg):
    length = random.randint(cfg.min_length, cfg.max_length)
    tokens = []
    indexes = []
    for _ in range(length):
        index = random.randint(0, len(ALL_TOKENS) - 1)
        # if len(indexes) and (indexes[-1] == index + 1):
        #     indexes.append(len(ALL_TOKENS) + 1)  # Insert Empty Space
        indexes.append(index + 1)
        tokens.append(ALL_TOKENS[index])
    return indexes, tokens


def prepare_directory(cfg):
    os.makedirs(cfg.data_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.data_dir, 'img'), exist_ok=True)

# ----- main starts here ------ #

cfg = Config(TEMPLATE_CFG)
cfg.parse_args()

prepare_directory(cfg)

image = ImageCaptcha()

with open(os.path.join(cfg.data_dir, 'data.txt'), 'w') as f:
    print(f'{cfg.max_length * 2}', file=f)
    for i in range(cfg.n_samples):
        indexes, tokens = gen_random_seq(cfg)
        img_dir = os.path.join(cfg.data_dir, 'img', f'{str(i).zfill(5)}.png')
        image.write(''.join(tokens), img_dir)
        indexes = [len(indexes)] + indexes
        indexes.extend([0 for _ in range(2 * cfg.max_length - len(indexes))])
        indexes = [str(x) for x in indexes]
        print('\t'.join(indexes) + '\t' + img_dir, file=f)
        print(f'Generated {i + 1:6d}/{cfg.n_samples:6d}', end='\r')
    print(f'Generated {cfg.n_samples:6d}/{cfg.n_samples:6d}')