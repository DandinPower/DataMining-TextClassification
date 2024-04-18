import os
import json 
import random

from argparse import ArgumentParser, Namespace
from abc import ABC, abstractmethod
from dataclasses import dataclass

from utils import text_preprocessing_pipeline

ONLY_TTITLE_AND_TEXT_FLAG = 'only_title_and_text'
CLEAN_ONLY_TTITLE_AND_TEXT_FLAG = 'clean_only_title_and_text'
MERGE_ALL_FEATURE_TO_TEXT_FLAG = 'merge_all_feature_to_text'
CLEAN_MERGE_ALL_FEATURE_TO_TEXT_FLAG = 'clean_merge_all_feature_to_text'

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
    
class CleanOnlyTitleAndText(ProcessedMethod):
    def process_train(self, data: Data) -> Data:
        title_text = text_preprocessing_pipeline(data.title)
        text_text = text_preprocessing_pipeline(data.text)
        data.processed_text = f'{title_text} [SEP] {text_text}'
        return data

    def process_test(self, data: Data) -> Data:
        title_text = text_preprocessing_pipeline(data.title)
        text_text = text_preprocessing_pipeline(data.text)
        data.processed_text = f'{title_text} [SEP] {text_text}'
        return data

class MergeAllFeatureToText(ProcessedMethod):
    def process_train(self, data: Data) -> Data:
        title_part = f'This Review Title is {data.title}'
        helpful_vote_part = f'{data.helpful_vote} people think this review is helpful'
        verified_purchase_part = 'This reviewer did purchase it' if data.verified_purchase else 'This reviewer did not purchase it'
        text_part = f'and the content is: {data.text}'
        data.processed_text = f'{title_part} {text_part}.There are other information for this review, one is {verified_purchase_part} and the other is {helpful_vote_part}.'
        return data

    def process_test(self, data: Data) -> Data:
        title_part = f'This Review Title is {data.title}'
        helpful_vote_part = f'{data.helpful_vote} people think this review is helpful'
        verified_purchase_part = 'This reviewer did purchase it' if data.verified_purchase else 'This reviewer did not purchase it'
        text_part = f'and the content is: {data.text}'
        data.processed_text = f'{title_part} {text_part}.There are other information for this review, one is {verified_purchase_part} and the other is {helpful_vote_part}.'
        return data
    
class CleanMergeAllFeatureToText(ProcessedMethod):
    def process_train(self, data: Data) -> Data:
        title_text = text_preprocessing_pipeline(data.title)
        text_text = text_preprocessing_pipeline(data.text)
        title_part = f'this review title is: {title_text}'
        helpful_vote_part = f'{data.helpful_vote} people think this review is helpful'
        verified_purchase_part = 'this reviewer did purchase it' if data.verified_purchase else 'this reviewer did not purchase it'
        text_part = f'and the content is: {text_text}'
        data.processed_text = f'{title_part} {text_part}.there are other information for this review, one is {verified_purchase_part} and the other is {helpful_vote_part}.'
        return data

    def process_test(self, data: Data) -> Data:
        title_text = text_preprocessing_pipeline(data.title)
        text_text = text_preprocessing_pipeline(data.text)
        title_part = f'this review title is: {title_text}'
        helpful_vote_part = f'{data.helpful_vote} people think this review is helpful'
        verified_purchase_part = 'this reviewer did purchase it' if data.verified_purchase else 'this reviewer did not purchase it'
        text_part = f'and the content is: {text_text}'
        data.processed_text = f'{title_part} {text_part}.there are other information for this review, one is {verified_purchase_part} and the other is {helpful_vote_part}.'
        return data

    
def get_processed_method(processed_method_flag: str) -> ProcessedMethod:
    if processed_method_flag == ONLY_TTITLE_AND_TEXT_FLAG:
        return OnlyTitleAndText()
    if processed_method_flag == MERGE_ALL_FEATURE_TO_TEXT_FLAG:
        return MergeAllFeatureToText()
    if processed_method_flag == CLEAN_ONLY_TTITLE_AND_TEXT_FLAG:
        return CleanOnlyTitleAndText()
    if processed_method_flag == CLEAN_MERGE_ALL_FEATURE_TO_TEXT_FLAG:
        return CleanMergeAllFeatureToText()
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
        f.write('index\ttext\tlabel\thelpful_vote\tverified_purchase')
        for d in data:
            f.write('\n')
            f.write(f'{d.index}\t{d.processed_text}\t{d.rating - 1}\t{d.helpful_vote}\t{d.verified_purchase}')

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
            rating=0,
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
    args_parser.add_argument('--processed_method', type=str, choices=[ONLY_TTITLE_AND_TEXT_FLAG, MERGE_ALL_FEATURE_TO_TEXT_FLAG, CLEAN_ONLY_TTITLE_AND_TEXT_FLAG, CLEAN_MERGE_ALL_FEATURE_TO_TEXT_FLAG], required=True)
    args = args_parser.parse_args()
    main(args)