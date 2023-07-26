TOKENIZER=gpt2
SIZE=all
PREFIX=128
IDS=
OUTPUT=gpt2_outputs/generate/yelp_sentiment0_${TOKENIZER}_${SIZE}_prefix${PREFIX}_ck${IDS}.txt 
MODEl=gpt2_outputs/yelp_sentiment0_${TOKENIZER}_${SIZE}_prefix${PREFIX}/checkpoint-${IDS}

python main_prefix_gpt2.py --task yelp_sentiment --mode generate --txt $OUTPUT \
--generate_num 100 --num_token_of_prefix 128 --prefix_model_path $MODEl \
--base_model_name gpt2

python main_classifier_roberta.py --task yelp_sentiment \
--mode eval_txt --data_path data/yelp_sentiment_roberta \
--model_path roberta_outputs/yelp_sentiment_classifier_roberta/checkpoint- \
--txt $OUTPUT

python main_classifier_roberta.py --task yelp_food_type3 \
--mode eval_txt --data_path data/yelp_food_roberta \
--model_path roberta_outputs/yelp_food_classifier_roberta/checkpoint- \
--txt $OUTPUT

python evaluation.py --txt $OUTPUT

python grammar_eval.py --txt $OUTPUT
