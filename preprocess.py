import random

from argparse import ArgumentParser, Namespace

from src.utils import set_seed, overwrite_folder, read_train_json, read_test_json, write_csv
from src.data import Data
from src.process_method import ProcessMethod, get_processed_method, get_choise_flag

def main(args: Namespace):
    set_seed(args.seed)
    overwrite_folder(args.output_dir)

    process_method: ProcessMethod = get_processed_method(args.processed_method)
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

    train_data = process_method.process_train_dataset(train_data)
    
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

    print(len(test_data))
    test_data = process_method.process_test_dataset(test_data)
    print(len(test_data))

    write_csv(test_data, args.output_dir, 'test.tsv')

if __name__ == "__main__":
    args_parser = ArgumentParser()
    args_parser.add_argument('--train_json', type=str, required=True)
    args_parser.add_argument('--test_json', type=str, required=True)
    args_parser.add_argument('--output_dir', type=str, required=True)
    args_parser.add_argument('--train_valid_ratio', type=float, required=True)
    args_parser.add_argument('--seed', type=int, default=42, required=True)
    args_parser.add_argument('--processed_method', type=str, choices=get_choise_flag(), required=True)
    args = args_parser.parse_args()
    main(args)