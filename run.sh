ORIGINAL_TRAIN_JSON="original_datasets/train.json"
ORIGINAL_TEST_JSON="original_datasets/test.json"
OUTPUT_DIR="hf_datasets"
TRAIN_VALID_RATIO=0.8
SEED=42

PROCESSED_METHOD="only_title_and_text"
# Available processed methods
# 1. only_title_and_text
# 2. clean_only_title_and_text
# 3. merge_all_feature_to_text
# 4. clean_merge_all_feature_to_text
# 5. only_12_star_only_title_and_text
# 6. only_45_star_only_title_and_text
# 7. group_12_and_45_only_title_and_text

UPLOAD_NAME="review_onlytitleandtext"
# Available upload names
# 1. review_onlytitleandtext
# 2. review_cleanonlytitleandtext
# 3. review_mergeallfeaturetotext
# 4. review_cleanmergeallfeaturetotext
# 5. review_only12staronlytitleandtext
# 6. review_only45staronlytitleandtext
# 7. review_group12and45onlytitleandtext

python preprocess.py \
    --train_json $ORIGINAL_TRAIN_JSON --test_json $ORIGINAL_TEST_JSON --output_dir $OUTPUT_DIR \
    --train_valid_ratio $TRAIN_VALID_RATIO --seed $SEED --processed_method $PROCESSED_METHOD

python create_datasets.py \
    --hf_folder $OUTPUT_DIR --upload_name $UPLOAD_NAME