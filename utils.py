import torch
from datasets import load_dataset
from transformers import GPT2TokenizerFast, CTRLTokenizer, BartTokenizerFast, Trainer, AutoTokenizer
import project_config

def get_init_prefix_weight(init_prefix, base_model_name, input_embeddings, num_token_of_prefix):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if init_prefix.endswith('unk'):
        sent = tokenizer.unk_token
    else:
        print("Error!")
        exit()
    input_ids = tokenizer(sent, return_tensors='pt').input_ids
    embs = input_embeddings(input_ids)[0]
    while embs.size(0) < num_token_of_prefix:
        embs = torch.cat((embs, embs), dim=0)
    embs = embs[-num_token_of_prefix:, :]
    return embs
    
def create_prompts(tokenizer_name, control_code=None, generate_eval=False):
    # different prompts for eval and test. we use eval prompts to select models
    prompts = ['Once upon a time', 'The book', 'The chicken', 'The city', 'The country', 'The house', 'The lake',
               'The last time', 'The movie', 'The painting', 'The pizza', 'The potato', 'The president of the country',
               'The road', 'The year is 1910.']#15
    if generate_eval:
        prompts = ['Anyone who wishes', 'As of', 'It is', 'In October 2014', 'While people can', 'Not only',
                   'The park', 'The game', 'The reward', 'The paper']#10
    tokenizer = None
    if tokenizer_name == 'gpt2':
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    elif tokenizer_name == 'bart':
        tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-base')
    if tokenizer_name == 'ctrl':
        assert control_code is not None, 'control_code can not be None'
        final_prompts = []
        for prompt in prompts:
            final_prompts.append(control_code + ' ' + prompt)
        prompts = final_prompts
    prompts_tensors = []
    for prompt in prompts:
        prompts_tensors.append(tokenizer(prompt, return_tensors='pt').input_ids)

    return prompts_tensors, prompts

def post_process_sent(sent, model, prompt=None):
    # post process generated text, keep the completed sentences
    sent = ''.join(reversed(sent))
    index = sent.find('.')
    if index > 0:
        sent = sent[index:]
    sent = ''.join(reversed(sent))
    if prompt is not None:
        sent = prompt + ' ' + sent
    return sent


def post_process_sents(sents, model='gpt2', prompt=None):
    # post process generated text, keep the completed sentences
    processed_sents = []
    for sent in sents:
        processed_sents.append(post_process_sent(sent, model, prompt))
    return processed_sents

def create_cla_dataset_from_txt(txt, task, tokenizer):
    assert txt is not None
    dataset = load_dataset('text.py', data_files={'test': txt})
    # max_eval_length = project_config.MAX_SST_LENGTH if task == 'sst' else project_config.MAX_AG_LENGTH
    max_eval_length = project_config.MAX_LENGTH
    dataset = dataset.map(lambda e: tokenizer(e['text'], truncation=True, max_length=max_eval_length,
                                              padding='max_length'), batched=True)
    return dataset['test']

def get_sorted_index(txt):
    if 'negative' in txt:
        return 0
    if 'positive' in txt:
        return 1
    if 'world' in txt:
        return 0
    if 'sports' in txt:
        return 1
    if 'business' in txt:
        return 2
    if 'sci' in txt:
        return 3
    return 0

