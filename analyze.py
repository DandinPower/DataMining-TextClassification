from argparse import ArgumentParser, Namespace
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

def draw_distribution(data, title, xlabel, ylabel, bins=50):
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(f'{title}.png')
    plt.clf()

def show_statistics(data, title):
    print(f'{title} Statistics:')
    print(f'Mean: {sum(data) / len(data)}')
    print(f'Min: {min(data)}')
    print(f'Max: {max(data)}')
    print(f'99.5% Percentile: {sorted(data)[int(len(data) * 0.995)]}')
    print(f'Median: {sorted(data)[len(data) // 2]}')

def analyze_dataset(dataset: Dataset, tokenizer: AutoTokenizer):
    '''
    The dataset must have column names 'text'.
    This function is used to analyze the dataset text length distribution and tokenized length distribution.
    '''
    text_lengths = [len(text) for text in dataset['text']]
    tokenized_lengths = [len(tokenizer(text)['input_ids']) for text in dataset['text']]

    draw_distribution(text_lengths, 'Text Length Distribution', 'Text Length', 'Frequency')
    show_statistics(text_lengths, 'Text Length')
    draw_distribution(tokenized_lengths, 'Tokenized Length Distribution', 'Tokenized Length', 'Frequency')
    show_statistics(tokenized_lengths, 'Tokenized Length')

def main(args: Namespace):
    dataset: Dataset = load_dataset(args.hf_folder, split=args.split)
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    analyze_dataset(dataset, tokenizer)
    
if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--hf_folder', type=str, required=True)
    arg_parser.add_argument('--tokenizer_name_or_path', type=str, required=True)
    arg_parser.add_argument('--split', type=str, required=True)
    args = arg_parser.parse_args()
    main(args)