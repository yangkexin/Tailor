import torch
from datasets import load_dataset
from transformers import GPT2TokenizerFast, CTRLTokenizer, BartTokenizerFast, Trainer, AutoTokenizer
import project_config

def get_init_prefix_weight(init_prefix, base_model_name, input_embeddings, num_token_of_prefix):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if init_prefix.endswith('unk'):
        sent = tokenizer.unk_token
    else:
        sent = project_config.init_words[init_prefix]
    input_ids = tokenizer(sent, return_tensors='pt').input_ids
    embs = input_embeddings(input_ids)[0]
    while embs.size(0) < num_token_of_prefix:
        embs = torch.cat((embs, embs), dim=0)
    embs = embs[-num_token_of_prefix:, :]
    return embs
