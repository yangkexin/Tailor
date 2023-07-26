from transformers import TrainingArguments, GPT2LMHeadModel, GPT2TokenizerFast, GPT2Config, Trainer, GPT2Model, AutoTokenizer,AutoModelForSequenceClassification
#for prompts
models = ['gpt2','gpt2-medium', 'gpt2-large']
for name in models:
    model = GPT2Model.from_pretrained(name)  
    tokenizer = AutoTokenizer.from_pretrained(name)
    model.save_pretrained("{}".format(name))
    tokenizer.save_pretrained("{}".format(name))
    print("save model to "+"{}".format(name))

#for sentiment classifer
roberta=AutoModelForSequenceClassification.from_pretrained("roberta-large", num_labels=2)
print(roberta)
roberta.save_pretrained("roberta-large-2")

#for food classifer
roberta=AutoModelForSequenceClassification.from_pretrained("roberta-large", num_labels=3)
print(roberta)
roberta.save_pretrained("roberta-large-3")



#for grammar eval
tokenizer = AutoTokenizer.from_pretrained("textattack/roberta-base-CoLA")
model = AutoModelForSequenceClassification.from_pretrained("textattack/roberta-base-CoLA")
tokenizer.save_pretrained("roberta-base-CoLA")
model.save_pretrained("roberta-base-CoLA")
