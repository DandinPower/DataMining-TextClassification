import os
import json 
import random

from argparse import ArgumentParser, Namespace
from abc import ABC, abstractmethod
from dataclasses import dataclass

ONLY_TTITLE_AND_TEXT_FLAG = 'only_title_and_text'
MERGE_ALL_FEATURE_TO_TEXT_FLAG = 'merge_all_feature_to_text'

def set_seed(seed: int):
    random.seed(seed)

@dataclass
class Data:
    index: str
    rating: int # 1-5 for train data, -1 for test data
    title: str
    text: str
    helpful_vote: int
    verified_purchase: bool 
    processed_text: str

    def __str__(self):
        return str(self.__dict__)            

class ProcessedMethod(ABC):
    @abstractmethod
    def process_train(self, data: Data) -> Data:
        pass

    @abstractmethod
    def process_test(self, data: Data) -> Data:
        pass

class OnlyTitleAndText(ProcessedMethod):
    def process_train(self, data: Data) -> Data:
        data.processed_text = f'Review Title: {data.title};Review Content: {data.text}'
        return data

    def process_test(self, data: Data) -> Data:
        data.processed_text = f'Review Title: {data.title};Review Content: {data.text}'
        return data

class MergeAllFeatureToText(ProcessedMethod):
    def process_train(self, data: Data) -> Data:
        title_part = f'Review Title: {data.title}'
        helpful_vote_part = f'How many people think this review is helpful: {data.helpful_vote}'
        verified_purchase_part = 'Verified Purchase: Yes' if data.verified_purchase else 'Verified Purchase: No'
        text_part = f'Review Content: {data.text}'
        data.processed_text = f'{helpful_vote_part};{verified_purchase_part};{title_part};{text_part}'
        return data

    def process_test(self, data: Data) -> Data:
        title_part = f'Review Title: {data.title}'
        helpful_vote_part = f'How many people think this review is helpful: {data.helpful_vote}'
        verified_purchase_part = 'Verified Purchase: Yes' if data.verified_purchase else 'Verified Purchase: No'
        text_part = f'Review Content: {data.text}'
        data.processed_text = f'{helpful_vote_part};{verified_purchase_part};{title_part};{text_part}'
        return data
    
def get_processed_method(processed_method_flag: str) -> ProcessedMethod:
    if processed_method_flag == ONLY_TTITLE_AND_TEXT_FLAG:
        return OnlyTitleAndText()
    if processed_method_flag == MERGE_ALL_FEATURE_TO_TEXT_FLAG:
        return MergeAllFeatureToText()
    raise ValueError(f'Invalid processed method flag: {processed_method_flag}')

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
        f.write('index\ttext\tlabel')
        for d in data:
            f.write('\n')
            f.write(f'{d.index}\t{d.processed_text}\t{d.rating}')

def main(args: Namespace):
    set_seed(args.seed)
    overwrite_folder(args.output_dir)

    process_method: ProcessedMethod = get_processed_method(args.processed_method)
    train_data: list[Data] = []
    temp_data = read_train_json(args.train_json)
    for index, d in enumerate(temp_data):
        train_data.append(Data(
            index=f'index_{index}',
            rating=int(d['rating']),
            title=d['title'],
            text=d['text'],
            helpful_vote=d['helpful_vote'],
            verified_purchase=d['verified_purchase'],
            processed_text=None
        ))

    train_data = list(map(process_method.process_train, train_data))
    
    random.shuffle(train_data)
    train_data_len = int(len(train_data) * args.train_valid_ratio)
    valid_data = train_data[:train_data_len]
    train_data = train_data[train_data_len:]

    write_csv(train_data, args.output_dir, 'train.tsv')
    write_csv(valid_data, args.output_dir, 'validation.tsv')

    test_data: list[Data] = []
    temp_data = read_test_json(args.test_json)
    for index, d in enumerate(temp_data):
        test_data.append(Data(
            index=f'index_{index}',
            rating=-1,
            title=d['title'],
            text=d['text'],
            helpful_vote=d['helpful_vote'],
            verified_purchase=d['verified_purchase'],
            processed_text=None
        ))

    test_data = list(map(process_method.process_test, test_data))
    write_csv(test_data, args.output_dir, 'test.tsv')

if __name__ == "__main__":
    args_parser = ArgumentParser()
    args_parser.add_argument('--train_json', type=str, required=True)
    args_parser.add_argument('--test_json', type=str, required=True)
    args_parser.add_argument('--output_dir', type=str, required=True)
    args_parser.add_argument('--train_valid_ratio', type=float, required=True)
    args_parser.add_argument('--seed', type=int, default=42, required=True)
    args_parser.add_argument('--processed_method', type=str, choices=[ONLY_TTITLE_AND_TEXT_FLAG, MERGE_ALL_FEATURE_TO_TEXT_FLAG], required=True)
    args = args_parser.parse_args()
    main(args)