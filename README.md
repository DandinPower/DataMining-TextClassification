# Data Mining Text Classification Dataset

It is a repository using product rating reviews to create a huggingface dataset for text classification task.

## Installation

1. Install the required packages
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

2. Setup Huggingface CLI
    ```bash
    pip install -U "huggingface_hub[cli]"
    huggingface-cli login # login by your WRITE token
    ```

## Usage

1. Setup `run.sh` config value and run the script
    ```bash
    bash run.sh
    ```

## Reminder

1. The label is getting by rating value, which is from 1 to 5. But the label is from 0 to 4.
2. If it is test dataset, the label is -1.

## Special Usage

1. You can run the following scripts to show the distribution of the dataset and tokenized dataset.

2. Modify the `analyze.sh` file.

3. Run the following command.
    ```bash
    bash analyze.sh
    ```

## Reference

1. [Huggingface Datasets](https://huggingface.co/docs/datasets/)
2. [Share Huggingface Datasets](https://huggingface.co/docs/datasets/share)