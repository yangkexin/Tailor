# !/opt/conda/bin/python3
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import random
from tqdm import tqdm
import numpy as np
from transformers import TrainingArguments, Trainer, GPT2TokenizerFast
from models.prefix_gpt2 import PrefixGPT2LMHeadModel, PrefixGPT2Config
from datasets import load_from_disk
import project_config
from utils import create_prompts, post_process_sents
import re
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(filename)s - %(lineno)s : %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_dataset(args):
    dataset = load_from_disk(args.data_path)
    return dataset['train'], dataset['validation'], dataset['validation']


def get_model(args):
    prefix_model_path, base_model_name, num_token_of_prefix, init_prefix = args.prefix_model_path, args.base_model_name, args.num_token_of_prefix, args.init_prefix
    if prefix_model_path is None:
        model_config = PrefixGPT2Config.from_pretrained(base_model_name, base_model_name=base_model_name,
                                                        num_token_of_prefix=num_token_of_prefix,
                                                        init_prefix=init_prefix)
        model = PrefixGPT2LMHeadModel(model_config)
        if args.mode=='train' and not init_prefix.endswith('random'):
            logging.info('init with method: {}'.format(init_prefix))
            model.init_prefix(init_prefix)
    else:
        model = PrefixGPT2LMHeadModel.from_pretrained(prefix_model_path)

    #save single attribute prompts
    # import pickle as pkl
    # path="/".join(args.weight_path.split("/")[:2])
    # if not os.path.isdir(path):
    #     os.makedirs(path)
    # pkl.dump(model.prefix_embed.weight.data,open(args.weight_path,"wb"))
    # exit()
    # return model

def get_model_concat(args):
    prefix_model_path, base_model_name, num_token_of_prefix, init_prefix = args.prefix_model_path, args.base_model_name, args.num_token_of_prefix, args.init_prefix

    model_config = PrefixGPT2Config.from_pretrained(base_model_name, base_model_name=base_model_name,
                                                    num_token_of_prefix=num_token_of_prefix,
                                                    init_prefix=init_prefix)
    model = PrefixGPT2LMHeadModel(model_config)
    logging.info("[Loading model1 from {}]".format(args.model1_path))
    logging.info("[Loading model2 from {}]".format(args.model2_path))
    model1 = PrefixGPT2LMHeadModel.from_pretrained(args.model1_path)
    model2 = PrefixGPT2LMHeadModel.from_pretrained(args.model2_path)
    prefix1 = model1.prefix_embed.weight.data
    prefix2 = model2.prefix_embed.weight.data
    prefix = torch.cat((prefix1, prefix2), dim=0) 
    print(prefix.size())
    print(prefix1.size())   
    model.prefix_embed.weight.data.copy_(prefix)
    return model


def generate_with_prompts(model, device, generate_eval=False, sents_per_prompt=1, samples_per_batch=10):
    #sents_per_prompt nums of generated sentences 
    model = model.to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained('checkpoints/gpt2')
    tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})
    prompts_tensors, _ = create_prompts('gpt2', generate_eval=generate_eval)
    sents_generated = []
    prompt_iter = tqdm(prompts_tensors, desc='generate')
    # import time
    # start_time = time.time() 
    for prompt in prompt_iter:
        if sents_per_prompt < samples_per_batch:
            samples_per_batch = sents_per_prompt
        num_batch = int(sents_per_prompt / samples_per_batch)

        for i in range(num_batch):
  
            output = model.generate(input_ids=prompt.repeat(samples_per_batch, 1).to(device),
                                    top_k=project_config.GENERATE_TOPK,
                                    do_sample=True,
                                    repetition_penalty=project_config.GENERATE_REPITATION_PALNITY,
                                    no_repeat_ngram_size=project_config.GENERATE_NO_REPEAT_NGRAM_SIZE,
                                    use_cache=True,
                                    min_length=project_config.GENERATE_MIN_LENGTH,
                                    max_length=project_config.GENERATE_MAX_LENGTH,
                                    pad_token_id=project_config.GPT2_PAD_TOKEN_ID)
            sents = tokenizer.batch_decode(output, skip_special_tokens=True)
            # sents_generated.extend(post_process_sents(sents))

            sents_generated.extend(sents)
    # end_time = time.time()
    # print('time cost', (end_time-start_time)/1500,'s')
    # exit()
    return sents_generated

def main(args):
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1
    args.device = device

    if args.mode.endswith('generate'):
        assert args.txt is not None
        model = get_model(args)
        sents_generated = generate_with_prompts(model, args.device, generate_eval=args.generate_eval,
                                                sents_per_prompt=args.generate_num)
        dir_name = '/'.join(args.txt.split('/')[:-1])
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        with open(args.txt, 'w', encoding='utf-8') as wf:
            for sent in sents_generated:
                sent = sent.replace('\n', ' ')
                # sent = process_final(sent)
                wf.write(sent + '\n')
        return 0

    if args.mode.endswith('generate_concat'):
        assert args.txt is not None
        assert args.model1_path is not None
        assert args.model2_path is not None
        model = get_model_concat(args)
        sents_generated = generate_with_prompts(model, args.device, generate_eval=args.generate_eval,
                                                sents_per_prompt=args.generate_num)
        dir_name = '/'.join(args.txt.split('/')[:-1])
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        with open(args.txt, 'w', encoding='utf-8') as wf:
            for sent in sents_generated:
                sent = sent.replace('\n', ' ')
                # sent = process_final(sent)
                wf.write(sent + '\n')
        return 0

    if args.mode.endswith('generate_dir'):
        assert args.txt is not None
        if not os.path.isdir(args.txt):
            os.makedirs(args.txt)
        dirs = os.listdir(args.prefix_model_path)
        root_path = args.prefix_model_path
        for dir in dirs:
            args.prefix_model_path = os.path.join(root_path, dir)
            model = get_model(args)
            suffix = str(dir.split('-')[-1]) + '.txt'
            txt = os.path.join(args.txt, suffix)
            sents_generated = generate_with_prompts(model, args.device, generate_eval=args.generate_eval,
                                                    sents_per_prompt=args.generate_num)
            with open(txt, 'w', encoding='utf-8') as wf:
                for sent in sents_generated:
                    sent = sent.replace('\n', ' ')
                    wf.write(sent + '\n')
        return 0

    model = get_model(args)
    train_dataset, dev_dataset, test_dataset = get_dataset(args)
  
    total_params = sum(p.numel() for n,p in model.named_parameters() if "transformer" in n)
    train_params = sum(x.numel() for x in filter(lambda y: y.requires_grad, model.parameters()))
    print("[Total Parameters: {}]".format(total_params))
    print("[Trainable Parameters: {}, Precentage: {}%]".format(train_params, train_params*100/total_params))

    evaluation_strategy = 'no' if args.no_eval else 'epoch'
    training_arguments = TrainingArguments(output_dir=args.output_dir,
                                           overwrite_output_dir=False,
                                           evaluation_strategy=evaluation_strategy,
                                           eval_steps=args.eval_steps,
                                           per_device_train_batch_size=args.per_device_train_batch_size,
                                           per_device_eval_batch_size=args.per_device_eval_batch_size,
                                           gradient_accumulation_steps=args.gradient_accumulation_steps,
                                           learning_rate=args.learning_rate,
                                           weight_decay=args.weight_decay,
                                           num_train_epochs=args.num_train_epochs,
                                           max_steps=args.max_steps,
                                           warmup_steps=args.warmup_steps,
                                           logging_dir=args.logging_dir,
                                           logging_first_step=True,
                                           logging_strategy='epoch',
                                           logging_steps=args.logging_steps,
                                           save_strategy='epoch',
                                           save_steps=args.save_steps,
                                           no_cuda=args.no_cuda,
                                           seed=args.seed,
                                           fp16=args.fp16,
                                           fp16_opt_level=args.fp16_opt_level,
                                           local_rank=args.local_rank,
                                           dataloader_drop_last=False,
                                           run_name=args.run_name,
                                           adam_epsilon=args.adam_epsilon,
                                           max_grad_norm=args.max_grad_norm,
                                           remove_unused_columns=True,
                                           ignore_data_skip=False,
                                           prediction_loss_only=False,
                                           dataloader_pin_memory=True,
                                           save_total_limit=30)
    set_seed(args)
    compute_metric = None
    trainer = Trainer(model=model,
                      args=training_arguments,
                      train_dataset=train_dataset,
                      eval_dataset=dev_dataset,
                      compute_metrics=compute_metric)

    if args.mode.endswith('train'):
        if args.local_rank in [-1, 0]:
            logging.info('***** training arguments *****')
            logging.info('warmup steps: {}'.format(args.warmup_steps))
            logging.info('lr: {}'.format(args.learning_rate))
            logging.info('fp16: {}'.format(args.fp16))
            logging.info('fp16-level: {}'.format(args.fp16_opt_level))
            logging.info('task: {}'.format(args.task))
            logging.info('prefix path: {}'.format(args.prefix_model_path))
            logging.info('base model: {}'.format(args.base_model_name))
        if args.prefix_model_path is not None:
            logging.info('training begin with checkpoint: {}'.format(args.prefix_model_path))
        trainer.train(resume_from_checkpoint=args.prefix_model_path)

    if args.mode.endswith('eval'):
        # predict and evaluate on dev_dataset
        trainer.evaluate()

    if args.mode.endswith('test'):
        trainer.evaluate(eval_dataset=test_dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1, help='local_rank for distributed training on gpus')
    parser.add_argument('--task', type=str, choices=['sst', 'ag_news', 'imdb','yelp_sentiment','yelp_food'],
                        help='the task for training and evaluating')
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'test', 'generate', 'generate_dir', 'generate_concat',"generate_keywords_single"],
                        default='train',
                        help='train,eval,test using dataset, eval_txt evaluating the sentences in txt')
    parser.add_argument('--txt', type=str, default=None, help='path of txt to write generated sentences')
    parser.add_argument('--data_path', type=str, default='./data', help='path of the dataset')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='dir to save checkpoints')
    parser.add_argument("--learning_rate", default=5e-5, type=float, help='the initial learning rate for Adam')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='number of updates steps to accumulate before performing a backward/update pass')
    parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay if we apply some')
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='epsilon for Adam optimizer')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='max gradient norm.')
    parser.add_argument('--num_train_epochs', default=20.0, type=float, help='number of training epochs to perform')
    parser.add_argument('--max_steps', default=-1, type=int,
                        help='If > 0: set total number of training steps to perform. Override num_epochs')
    parser.add_argument('--warmup_steps', default=00, type=int, help='linear warmup over warmup_steps')
    parser.add_argument('--logging_steps', type=int, default=500, help='log every X updates steps')
    parser.add_argument('--save_steps', type=int, default=500, help='Save checkpoint every X updates steps')
    parser.add_argument('--eval_steps', type=int, default=500, help='evaluate every X updates steps')
    parser.add_argument('--no_eval', action='store_true', help='train without eval on validation set if set')
    parser.add_argument('--fp16', action='store_true', help='whether to use fp16')
    parser.add_argument('--fp16_opt_level', choices=['O0', 'O1', 'O2', 'O3'], default='O0',
                        help='For fp16: Apex AMP optimization level selected in ["O0", "O1", "O2", and "O3"]')
    parser.add_argument('--no_cuda', action='store_true', help='training on cpu if set')
    parser.add_argument('--seed', type=int, default=42, help='random seed for initialization')
    parser.add_argument('--per_device_train_batch_size', default=16, type=int,
                        help='batch size per GPU/CPU for training')
    parser.add_argument('--per_device_eval_batch_size', default=32, type=int,
                        help='batch size per GPU/CPU for evaluating')
    parser.add_argument('--logging_dir', default='./logs', type=str, help='dir to save logs')
    parser.add_argument('--run_name', type=str, default='run_1', help='trainer run name')
    parser.add_argument('--datasize', type=float, default=1.0,
                        help='how much data used in training, e.g. 0.5 means half.')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='whether or not to use gradient checkpointing')
    parser.add_argument('--prefix_model_path', type=str, default=None,
                        help='checkpoint path of prefix to load for continuing-training or evaluate, init for training')
    parser.add_argument('--num_token_of_prefix', type=int, required=True, help='number of tokens of prefix, <= 256')
    parser.add_argument('--generate_num', type=int, default=100, help='number of sentences to generate (for each prompt)')
    parser.add_argument('--base_model_name', type=str, default='gpt2', help='pretrained gpt2 model to load',
                        choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'])
    parser.add_argument('--init_prefix', type=str, default='random',
                        help='method to init the prefix embedding. must be in format "task_attr_method", e.g. "sst_negative_random", the method should be one of [random, label, unk, keyword, keywords]')
    parser.add_argument('--generate_eval', action='store_true', help='whether generate with eval prompts')
    parser.add_argument('--model1_path', type=str, default="", help='model1 path for concating in generating)')
    parser.add_argument('--model2_path', type=str, default="", help='model2 path for concating in generating)')   
    parser.add_argument('--weight_path', type=str, default="", help='single attribute weight save path)')  
    args = parser.parse_args()
    main(args)
