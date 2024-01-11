# Tailor
### Step 0 Installation
See requirements.txt

### Step 1 Download pre-trained models

```
python checkpoints/download_models.py
```


### Step 2 Process dataset

&emsp;&emsp;See readme.md in the data folder

### Step 3 Train the single-attribute prompts/classifier
#### 1. Single-attribute prompts training
&emsp;&emsp;Using train_single_attribute_prompt_sentiment.sh/train_single_attribute_prompt_topic.sh
#### 2. Single-attribute classifier training
&emsp;&emsp;Using train_food_classifier.sh/train_sentiment_classifier.sh 
#### 3. Single-attribute classifier testing
&emsp;&emsp;Using test_food_classifier.sh/test_sentiment_classifier.sh
#### 4. Generating single-attribute sentences from given prompts and then evaluating the generating results
&emsp;&emsp;Using test_single_attribute_prompt_sentiment.sh/test_single_attribute_prompt_topic.sh
   
### Step 4 Preprocessing single-attribute data/prompts for multi-attribute text generation task
#### 1. Using attribute classifiers to pseudo-label single-attribute data
&emsp;&emsp;As we mentioned in the paper, each single attribute data sample needs to be labeled again, such as the food single attribute text will get the annotation about the sentiment attribute. Here, we use the attribute classifier trained in the previous step to score the single attribute training data:
   - Before starting the annotation, remove the attribute labels for each line in the training dataset file (i.e., each line of the annotation file is a sentence, not a label + space + sentence). Save the file as "train_yelp_sentiment_prompt_without_label.txt" and "train_yelp_food_prompt_without_label.txt".
   - After that, we annotate the above two files separately using label_food_data.sh/label_sentiment_data.sh
   
   
