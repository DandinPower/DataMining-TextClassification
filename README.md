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

## Reference

1. [Huggingface Datasets](https://huggingface.co/docs/datasets/)
2. [Share Huggingface Datasets](https://huggingface.co/docs/datasets/share)