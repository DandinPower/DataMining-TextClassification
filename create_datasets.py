from argparse import ArgumentParser, Namespace
from datasets import load_dataset, Dataset

def main(args: Namespace):
    dataset: Dataset = load_dataset(args.hf_folder)
    dataset.push_to_hub(args.upload_name)
    
if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--hf_folder', type=str, required=True)
    arg_parser.add_argument('--upload_name', type=str, required=True)
    args = arg_parser.parse_args()
    main(args)