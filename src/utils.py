import os
import json 
import random

from .data import Data

def set_seed(seed: int):
    random.seed(seed)

def read_train_json(json_path: str):
    with open(json_path) as f:
        data = json.load(f)
    return data

def read_test_json(json_path: str):
    with open(json_path) as f:
        data = json.load(f)
    return data

def overwrite_folder(output_dir: str):
    if os.path.exists(output_dir):
        os.system(f'rm -rf {output_dir}')
    os.makedirs(output_dir)

def write_csv(data: list[Data], output_dir: str, file_name: str):
    output_dir = os.path.join(output_dir, 'data')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, file_name)

    if os.path.exists(output_path):
        os.system(f'rm -f {output_path}')

    with open(output_path, 'w') as f:
        f.write('index\ttext\tlabel\thelpful_vote\tverified_purchase')
        for d in data:
            f.write('\n')
            f.write(f'{d.index}\t{d.processed_text}\t{d.rating}\t{d.helpful_vote}\t{d.verified_purchase}')