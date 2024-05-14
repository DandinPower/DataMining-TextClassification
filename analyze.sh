OUTPUT_DIR="hf_datasets"
SPLIT_NAME="train"
TOKENIZER_NAME="microsoft/deberta-v3-base"

python analyze.py \
    --hf_folder $OUTPUT_DIR --split $SPLIT_NAME --tokenizer_name_or_path $TOKENIZER_NAME