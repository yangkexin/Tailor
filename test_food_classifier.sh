DATA=data/yelp_food_type3_roberta/
OUTPUT==roberta_outputs/yelp_food_classifier_roberta
LOGS=roberta_logs/yelp_food_classifier_roberta
CHECKPOINT=roberta_outputs/yelp_food_classifier_roberta/checkpoint-/

# python main_classifier_roberta.py --task yelp_food_type3 --mode eval --data_path $DATA \
# --output_dir $OUTPUT --logging_dir $LOGS \
# --model_path $CHECKPOINT

python main_classifier_roberta.py --task yelp_food_type3 --mode test --data_path $DATA \
--output_dir $OUTPUT --logging_dir $LOGS \
--model_path $CHECKPOINT \
--per_device_eval_batch_size 128
