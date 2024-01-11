TOKENIZER=gpt2
SIZE=all
PREFIX=128
IDS=
# label 0 for negative sentiment/ 1 for positive sentiment
LABEL=0
OUTPUT=gpt2_outputs/generate/yelp_sentiment${LABEL}$_${TOKENIZER}_${SIZE}_prefix${PREFIX}_ck${IDS}.txt 
MODEl=gpt2_outputs/yelp_sentiment${LABEL}$_${TOKENIZER}_${SIZE}_prefix${PREFIX}/checkpoint-${IDS}
WEIGHT=yelp_sentiment_food_prompt/yelp_sentiment${LABEL}$.pkl
python main_prefix_gpt2.py --task yelp_sentiment --mode generate --txt $OUTPUT \
--generate_num 100 --num_token_of_prefix 128 --prefix_model_path $MODEl \
--base_model_name gpt2 --weight_path $WEIGHT \
