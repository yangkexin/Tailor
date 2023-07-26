# !/opt/conda/bin/python3
# -*- coding: utf-8 -*-

import argparse
from datasets import load_dataset
from transformers import RobertaTokenizerFast
import sys
sys.path.append("..")
import project_config

def process_yelp_sentiment(save_path):
    dataset = load_dataset("yelp_sentiment/yelp_classifier.py",
    data_files="yelp_sentiment/")
    dataset = tokenize_yelp_sentiment(dataset)
    dataset.save_to_disk(save_path)


def tokenize_yelp_sentiment(dataset):
    def create_label(example):
        example['labels'] = int(example['label'])

    dataset = dataset.map(create_label, remove_columns=['label'])
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    dataset = dataset.map(lambda example: tokenizer(example['text'], truncation=True, max_length=project_config.MAX_AG_LENGTH,
                                                    padding='max_length'), batched=True)
    return dataset


def process_yelp_food(save_path):
    dataset = load_dataset("yelp_food/yelp_classifier.py",
    data_files="yelp_food/")
    dataset = tokenize_yelp_food(dataset)
    dataset.save_to_disk(save_path)


def tokenize_yelp_food(dataset):
    def create_label(example):
        example['labels'] = int(example['label'])

    dataset = dataset.map(create_label, remove_columns=['label'])
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    dataset = dataset.map(lambda example: tokenizer(example['text'], truncation=True, max_length=project_config.MAX_AG_LENGTH,
                                                    padding='max_length'), batched=True)
    return dataset


def main(args):

    if args.dataset == 'yelp_sentiment':
        process_yelp_sentiment(args.save_path)

    if args.dataset == 'yelp_food':
        process_yelp_food(args.save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='./data', help='dir path to save processed data')
    parser.add_argument('--dataset', type=str, choices=['yelp_sentiment','yelp_food'], help='dataset')
    args = parser.parse_args()
    main(args)
