ORIGINAL_TRAIN_JSON="original_datasets/train.json"
ORIGINAL_TEST_JSON="original_datasets/test.json"
OUTPUT_DIR="hf_datasets"
TRAIN_VALID_RATIO=0.8
SEED=42
# PROCESSED_METHOD="only_title_and_text"
# PROCESSED_METHOD="clean_only_title_and_text"
# PROCESSED_METHOD="merge_all_feature_to_text"
PROCESSED_METHOD="clean_merge_all_feature_to_text"

# UPLOAD_NAME="DataMining-TextClassification-OnlyTitleAndText"
# UPLOAD_NAME="DataMining-TextClassification-CleanOnlyTitleAndText"
# UPLOAD_NAME="DandinPower/DataMining-TextClassification-MergeAllFeatureToText"
UPLOAD_NAME="DandinPower/DataMining-TextClassification-CleanMergeAllFeatureToText"


python preprocess.py \
    --train_json $ORIGINAL_TRAIN_JSON --test_json $ORIGINAL_TEST_JSON --output_dir $OUTPUT_DIR \
    --train_valid_ratio $TRAIN_VALID_RATIO --seed $SEED --processed_method $PROCESSED_METHOD

python create_datasets.py \
    --hf_folder $OUTPUT_DIR --upload_name $UPLOAD_NAME