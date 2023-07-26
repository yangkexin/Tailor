DATA=data/yelp_food_roberta/
OUTPUT==roberta_outputs/yelp_food_classifier_roberta
LOGS=roberta_logs/yelp_food_classifier_roberta
MODEL=checkpoints/roberta-large-3

python main_classifier_roberta.py --task yelp_food_type3 --mode train --data_path $DATA \
--save_steps 500 --logging_steps 500 --eval_steps 500 \
--per_device_train_batch_size 8 \
--gradient_accumulation_steps 8 --run_name yelp_food_type3_v2_classifier_roberta \
--learning_rate 1e-5 --weight_decay 0.01 --warmup_steps 800 \
--output_dir $OUTPUT --logging_dir $LOGS \
--model_path $MODEL --num_train_epochs 20
