python main_classifier_roberta.py --task yelp_sentiment \
--mode eval_txt --data_path data/yelp_sentiment_roberta \
--model_path ##fill in with your path## \
--txt data/yelp_food/test_yelp_food_prompt_without_label.txt \
--save_dir /mnt/nas/users/yangkexin.ykx/prompt/PretrainPrefix_v1/data/yelp_food/test_yelp_food_prompt_with_sentimentabel.txt
