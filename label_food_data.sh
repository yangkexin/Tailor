python main_classifier_roberta.py --task yelp_food_type3 \
--mode eval_txt --data_path data/yelp_food_type3_roberta \
--model_path #need to fill with your path# \
--txt data/yelp_sentiment/test_yelp_sentiment_prompt_without_label.txt \
--save_dir data/yelp_sentiment/test_yelp_sentiment_prompt_with_foodlabel.txt
