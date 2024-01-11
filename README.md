# Tailor
### Step 0 Installation
See requirements.txt

### Step 1 Download pre-trained models

```
python checkpoints/download_models.py
```


### Step 2 Process dataset

See readme.md in the data folder

### Step 3 Train the single prompts
1. Using train_single_attribute_prompt_sentiment.sh/train_single_attribute_prompt_topic.sh for single prompts training
2. Using test_single_attribute_prompt_sentiment.sh/test_single_attribute_prompt_topic.sh for generating sentences from given prompts


### Step 4 Train the attribute classifiers
