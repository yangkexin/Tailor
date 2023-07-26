SIZE=all
PREFIX=128
RUN=yelp_sentiment0_${TOKENIZER}_${SIZE}_${INIT}_prefix${PREFIX}

DATA=data/yelp_sentiment_gpt2/negative
OUTPUT=gpt2_outputs/yelp_sentiment0_${TOKENIZER}_${SIZE}_prefix${PREFIX}
LOGS=gpt2_logs/yelp_sentiment0_${TOKENIZER}_${SIZE}_prefix${PREFIX}

python main_prefix_gpt2.py --task yelp_sentiment --mode train --data_path $DATA \
--per_device_train_batch_size 16 \
--run_name $RUN --learning_rate 5e-3 --num_token_of_prefix $PREFIX \
--output_dir $OUTPUT --logging_dir $LOGS \
--base_model_name $TOKENIZER
