## Usage
### Download the classifier dataset
See https://drive.google.com/file/d/19Q5ujkELaZrw2xh-H-XehJQQbF53VU4O/view?usp=drive_link
### Process dataset for single attribute prompts
```
python process_data_gpt2.py --dataset yelp_sentiment --save_path yelp_sentiment_gpt2/
```
### Process dataset for single attribute classifiers
```
python process_data_classification.py --dataset yelp_sentiment --save_path yelp_sentiment_roberta/
```
