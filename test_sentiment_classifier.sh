DATA=data/yelp_sentiment_roberta/
OUTPUT=roberta_outputs/yelp_sentiment_classifier_roberta
LOGS=roberta_logs/yelp_sentiment_classifier_roberta
CHECKPOINT=roberta_outputs/yelp_sentiment_classifier_roberta/checkpoint-4500/

python main_classifier_roberta.py --task yelp_sentiment --mode eval --data_path $DATA \
--output_dir $OUTPUT --logging_dir $LOGS \
--model_path $CHECKPOINT

python main_classifier_roberta.py --task yelp_sentiment --mode test --data_path $DATA \
--output_dir $OUTPUT --logging_dir $LOGS \
--model_path $CHECKPOINT
