# !/opt/conda/bin/python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from datasets import load_dataset
from transformers import GPT2TokenizerFast
import os
import sys
import pickle

sys.path.append("..")
import project_config


def process_yelp_sentiment(save_path, size):
    dataset = load_dataset("yelp_sentiment/yelp_prompt.py",
    data_files="yelp_sentiment/")
    if size < 1:
        dataset['train'] = dataset['train'].train_test_split(test_size=size)['test']
    dataset = tokenize_yelp_sentiment(dataset)
    save_paths = [os.path.join(save_path, 'negative'),
                  os.path.join(save_path, 'positive')                
                  ]
    for i in range(len(save_paths)):
        split_dataset = dataset.filter(lambda e: e['topic'] == i)
        split_dataset.save_to_disk(save_paths[i])

def tokenize_yelp_sentiment(dataset):
    # train_dataset = dataset['train']
    # split_dataset = train_dataset.train_test_split(0.02)
    # dataset['train'] = split_dataset['train']
    # dataset['validation'] = split_dataset['test']

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})
    dataset = dataset.map(
        lambda example: tokenizer(example['text'], truncation=True, max_length=project_config.MAX_AG_LENGTH,
                                  padding='max_length'), batched=True)

    def create_topic_label(example):
        example['topic'] = int(example['label'])
        input_arr = np.array(example['input_ids'])
        mask_arr = np.array(example['attention_mask'])
        np.place(input_arr, mask_arr == 0, -100)
        example['labels'] = input_arr.tolist()
        return example

    dataset = dataset.map(create_topic_label, remove_columns=['label'])
    return dataset

def process_yelp_food(save_path, size):
    dataset = load_dataset("yelp_food/yelp_prompt.py",
    data_files="yelp_food/")
    if size < 1:
        dataset['train'] = dataset['train'].train_test_split(test_size=size)['test']
    dataset = tokenize_yelp_food(dataset)
    save_paths = [os.path.join(save_path, 'mexican'),
                  os.path.join(save_path, 'american'),
                  os.path.join(save_path, 'asian'),                
                  ]
    for i in range(len(save_paths)):
        split_dataset = dataset.filter(lambda e: e['topic'] == i)
        split_dataset.save_to_disk(save_paths[i])

def tokenize_yelp_food(dataset):
    # train_dataset = dataset['train']
    # split_dataset = train_dataset.train_test_split(0.02)
    # dataset['train'] = split_dataset['train']
    # dataset['validation'] = split_dataset['test']

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})
    dataset = dataset.map(
        lambda example: tokenizer(example['text'], truncation=True, max_length=project_config.MAX_AG_LENGTH,
                                  padding='max_length'), batched=True)

    def create_topic_label(example):
        example['topic'] = int(example['label'])
        input_arr = np.array(example['input_ids'])
        mask_arr = np.array(example['attention_mask'])
        np.place(input_arr, mask_arr == 0, -100)
        example['labels'] = input_arr.tolist()
        return example

    dataset = dataset.map(create_topic_label, remove_columns=['label'])
    return dataset


def process_yelp_connector(save_path, size):
    dataset = load_dataset("yelp_connector/connector.py",
    data_files="yelp_connector/")
    if size < 1:
        dataset['train'] = dataset['train'].train_test_split(test_size=size)['test']
    dataset = tokenize_yelp_connector(dataset)
    dataset.save_to_disk(save_path)

def tokenize_yelp_connector(dataset):
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})
    dataset = dataset.map(
        lambda example: tokenizer(example['text'], truncation=True, max_length=project_config.MAX_AG_LENGTH,
                                  padding='max_length'), batched=True)

    def create_topic_label(example):
        example['topic'] = int(example['label'])
        input_arr = np.array(example['input_ids'])
        mask_arr = np.array(example['attention_mask'])
        np.place(input_arr, mask_arr == 0, -100)
        example['labels'] = input_arr.tolist()
        return example

    dataset = dataset.map(create_topic_label, remove_columns=['label'])
    return dataset


def process_yelp_connector_argmax(save_path, size):
    dataset = load_dataset("yelp_connector/yelp_connector.py",
    data_files="yelp_connector/")
    if size < 1:
        dataset['train'] = dataset['train'].train_test_split(test_size=size)['test']
    dataset = tokenize_yelp_connector_argmax(dataset)
    dataset.save_to_disk(save_path)

def tokenize_yelp_connector_argmax(dataset):
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})
    dataset = dataset.map(
        lambda example: tokenizer(example['text'], truncation=True, max_length=project_config.MAX_LENGTH,
                                  padding='max_length'), batched=True)
    task_embed = pickle.load(open("yelp_prompt_weight/single_task_embedding_dict.pkl","rb"))

    def create_topic_label(example):
        # for food [predict sentiment label, negative probability, positive probability, placeholder(-1.0), food label]
        # for sentiment [predict food label, mexican probability, american probability, asian probability, sentiment label]
        temp_list = [float(line) for line in example['label'].split("\t")]
        task_lable = int(temp_list[-1])
        fake_lable = int(temp_list[0])
        if task_lable in [2,3,4]:
            embed1 = task_embed[task_lable]
            embed2 = task_embed[fake_lable]
            final_embeds = embed2.tolist() + embed1.tolist() 
        else:
            embed1 = task_embed[task_lable]
            embed2 = task_embed[fake_lable+2]   
            final_embeds = embed1.tolist() + embed2.tolist()      
        example['topic'] = final_embeds
        input_arr = np.array(example['input_ids'])
        mask_arr = np.array(example['attention_mask'])
        np.place(input_arr, mask_arr == 0, -100)
        example['labels'] = input_arr.tolist()
        return example

    dataset = dataset.map(create_topic_label, remove_columns=['label'])
    return dataset


def process_yelp_connector_weighted(save_path, size):
    dataset = load_dataset("yelp_connector/yelp_connector.py",
    data_files="yelp_connector/")
    if size < 1:
        dataset['train'] = dataset['train'].train_test_split(test_size=size)['test']
    dataset = tokenize_yelp_connector_weighted(dataset)
    dataset.save_to_disk(save_path)

def tokenize_yelp_connector_weighted(dataset):
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})
    dataset = dataset.map(
        lambda example: tokenizer(example['text'], truncation=True, max_length=project_config.MAX_LENGTH,
                                  padding='max_length'), batched=True)
    task_embed = pickle.load(open("yelp_prompt_weight/single_task_embedding_dict.pkl","rb"))

    def create_topic_label(example):
        # for food [predict sentiment label, negative probability, positive probability, placeholder(-1.0), food label]
        # for sentiment [predict food label, mexican probability, american probability, asian probability, sentiment label]
        temp_list = [float(line) for line in example['label'].split("\t")]
        task_lable = int(temp_list[-1])
        if task_lable in [2,3,4]:
            fake_prob = temp_list[1:3]
            embed1 = task_embed[task_lable]
            embed2 = task_embed[0]*fake_prob[0]+task_embed[1]*fake_prob[1]
            final_embeds = embed2.tolist() + embed1.tolist() 
        else:
            fake_prob = temp_list[1:4]
            embed1 = task_embed[task_lable]
            embed2 = task_embed[2]*fake_prob[0]+task_embed[3]*fake_prob[1]+task_embed[4]*fake_prob[2]    
            final_embeds = embed1.tolist() + embed2.tolist()      
        example['topic'] = final_embeds
        input_arr = np.array(example['input_ids'])
        mask_arr = np.array(example['attention_mask'])
        np.place(input_arr, mask_arr == 0, -100)
        example['labels'] = input_arr.tolist()
        return example

    dataset = dataset.map(create_topic_label, remove_columns=['label'])
    return dataset


def main(args):
    assert 0.0 < args.size <= 1.0
    if args.dataset == 'yelp_food':
        process_yelp_food(args.save_path, args.size)
    if args.dataset == 'yelp_sentiment':
        process_yelp_sentiment(args.save_path, args.size)
    if args.dataset == "yelp_connector":
        process_yelp_connector(args.save_path, args.size)
    if args.dataset == "yelp_connector_argmax":
        process_yelp_connector_argmax(args.save_path, args.size)
    if args.dataset == "yelp_connector_weighted":
        process_yelp_connector_weighted(args.save_path, args.size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, help='path to save processed data')
    parser.add_argument('--dataset', type=str, choices=[
    "yelp_food","yelp_sentiment","yelp_connector","yelp_connector_argmax","yelp_connector_weighted"], help='dataset')
    parser.add_argument('--size', type=float, default=1.0,
                        help='how many samples to process and save, e.g. 0.5 means half of dataset will be processed and saves')
    args = parser.parse_args()
    main(args)
