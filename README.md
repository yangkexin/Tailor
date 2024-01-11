# Tailor
### Step 0 Installation
See requirements.txt

### Step 1 Download pre-trained models

```
python checkpoints/download_models.py
```


### Step 2 Process dataset

See readme.md in the data folder

### Step 3 Train the single-attribute prompts/classifier
1. Using train_single_attribute_prompt_sentiment.sh/train_single_attribute_prompt_topic.sh for single prompts training
2. Using train_food_classifier.sh/train_sentiment_classifier.sh to train the attribute evaluators, and you can use test_food_classifier.sh/test_sentiment_classifier.sh to evaluate the classifiers
3. Using test_single_attribute_prompt_sentiment.sh/test_single_attribute_prompt_topic.sh for generating sentences from given prompts
   
### Step 4 Preprocessing single-attribute data/prompts for multi-attribute text generation task
#### 1. Using attribute classifiers to pseudo-label single-attribute data
   As we mentioned in the paper, each single attribute data needs to be labeled again, such as the food single attribute text will get the annotation about the emotional attribute. Here, we use the attribute classifier trained in the previous step to score the single attribute training data. Before starting the annotation, remove the attribute labels for each line in the training dataset file (i.e., each line of the annotation file is a sentence, not a label + space + sentence). Save the file as "train_yelp_sentiment_prompt_without_label.txt" and "train_yelp_food_prompt_without_label.txt".
   
   After that, we annotate the above two files separately using label_food_data.sh/label_sentiment_data.sh
   
   
