
from argparse import ArgumentParser
import math
import string

from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForSequenceClassification


def grammaticality(sentences, tokenizer, model, device='cuda'):
    with torch.no_grad():
        total_good = 0
        for sent in tqdm(sentences, total=len(sentences)):
            good_prob = F.softmax(model(tokenizer.encode(sent, return_tensors='pt').to(device))[0].flatten(), dim=0)[1]
            total_good += good_prob
        return total_good / len(sentences) # avg probability of grammaticality according to model

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--txt', type=str)
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    args = parser.parse_args()

    preds = []
    with open(args.txt, 'r') as rf:
        for line in rf:
            preds.append(line.strip()) # drop \n but not beginning spaces if any
   
    grammar_tokenizer = AutoTokenizer.from_pretrained('checkpoints/roberta-base-CoLA')
    grammar_model = AutoModelForSequenceClassification.from_pretrained('checkpoints/roberta-base-CoLA').to(args.device)
    grammar_model.eval()
    print('grammaticality', grammaticality(preds, grammar_tokenizer, grammar_model, device=args.device))
