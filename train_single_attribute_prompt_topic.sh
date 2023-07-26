TOKENIZER=gpt2
SIZE=all
PREFIX=128
RUN=yelp_filter_food0_${TOKENIZER}_${SIZE}_${INIT}_prefix${PREFIX}

DATA=data/yelp_food_gpt2/mexican
OUTPUT=gpt2_outputs/yelp_food0_${TOKENIZER}_${SIZE}_prefix${PREFIX}
LOGS=gpt2_logs/yelp_food0_${TOKENIZER}_${SIZE}_prefix${PREFIX}

python main_prefix_gpt2.py --task yelp_food --mode train --data_path $DATA \
--per_device_train_batch_size 16 \
--run_name $RUN --learning_rate 5e-3 --num_token_of_prefix $PREFIX \
--output_dir $OUTPUT --logging_dir $LOGS \
--base_model_name $TOKENIZER
