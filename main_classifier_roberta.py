# !/opt/conda/bin/python3
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import random
import numpy as np
from transformers import TrainingArguments, RobertaForSequenceClassification, RobertaTokenizerFast, Trainer
from datasets import load_from_disk
from utils import create_cla_dataset_from_txt, get_sorted_index


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_dataset(args):
    dataset = load_from_disk(args.data_path)
    # split_dataset = dataset['train'].train_test_split(0.0001)
    # print(split_dataset["test"][10])
    # return split_dataset['test'], split_dataset['test'], split_dataset['test']
    return dataset['train'], dataset['validation'], dataset['test']


def get_model(args):
    assert args.task in ["yelp_sentiment","yelp_food_type3"]
    if args.task in ["yelp_sentiment"]:
        if "roberta-large" in args.model_path:
            model = RobertaForSequenceClassification.from_pretrained("checkpoints/roberta-large-2", num_labels=2)
        else:
            model = RobertaForSequenceClassification.from_pretrained(args.model_path, num_labels=2)

    if args.task in ["yelp_food_type3"]:
        if "roberta-large" in args.model_path:
            model = RobertaForSequenceClassification.from_pretrained("checkpoints/roberta-large-3", num_labels=3)
        else:
            model = RobertaForSequenceClassification.from_pretrained(args.model_path, num_labels=3)

    return model


def compute_metric_sst(eval_prediction):
    predictions = np.array(eval_prediction.predictions)
    label_ids = np.array(eval_prediction.label_ids)
    predictions_ids = np.argmax(predictions, axis=1)
    total_num = len(label_ids)
    y_true = label_ids
    y_pred = predictions_ids
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
    print("Precision: {:.3f}".format(precision_score(y_true, y_pred, average='macro')))     
    print("Recall: {:.3f}".format(recall_score(y_true, y_pred, average='macro'))) 
    print("F1: {:.3f}".format(f1_score(y_true, y_pred, average='macro')))
    print("confusion_matrix:")
    print(confusion_matrix(y_true, y_pred))
    # print("label ids:",label_ids)
    # print("total nums:",total_num)
    acc = sum((predictions_ids - label_ids) == 0) / total_num
    label_total = [0.0, 0.0]
    label_correct = [0.0, 0.0]
    for i in range(total_num):
        label_total[label_ids[i]] += 1.0
        if predictions_ids[i] == label_ids[i]:
            label_correct[label_ids[i]] += 1.0
    label_acc = np.array(label_correct) / np.array(label_total)
    print(label_total)
    print(label_correct)
    print('acc: ', acc)
    print('label_0_acc: {:.2f}'.format(label_acc[0]))
    print('label_1_acc: {:.2f}'.format(label_acc[1]))
    return {'acc': acc}


def compute_metric(eval_prediction):
    predictions = np.array(eval_prediction.predictions)
    label_ids = np.array(eval_prediction.label_ids)
    predictions_ids = np.argmax(predictions, axis=1)
    y_true = label_ids
    y_pred = predictions_ids

    with open("yelp_food_score_all_train_data.txt","a") as f:
        for i in range(len(predictions_ids)):
            temp_id = str(predictions_ids[i])
            temp_prediction = "\t".join([str(line) for line in predictions[i]])
            f.write(str(label_ids[i])+"\t"+temp_id+"\t"+temp_prediction+"\n")
    f.close()
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
    print("Precision: {:.3f}".format(precision_score(y_true, y_pred, average='macro')))     
    print("Recall: {:.3f}".format(recall_score(y_true, y_pred, average='macro'))) 
    print("F1: {:.3f}".format(f1_score(y_true, y_pred, average='macro')))
    print("confusion_matrix:")
    print(confusion_matrix(y_true, y_pred))

    total_num = len(label_ids)
    acc = sum((predictions_ids - label_ids) == 0) / total_num
    label_total = [0.0, 0.0, 0.0]
    label_correct = [0.0, 0.0, 0.0]
    for i in range(total_num):
        label_total[label_ids[i]] += 1.0
        if predictions_ids[i] == label_ids[i]:
            label_correct[label_ids[i]] += 1.0
    label_acc = np.array(label_correct) / np.array(label_total)
    print(label_total)
    print(label_correct)
    print('acc: ', acc)
    print('label_0_acc: {:.2f}'.format(label_acc[0]))
    print('label_1_acc: {:.2f}'.format(label_acc[1]))
    print('label_2_acc: {:.2f}'.format(label_acc[2]))
    return {'acc': acc}



def main(args):
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1
    args.device = device

    train_dataset, dev_dataset, test_dataset = get_dataset(args)
    print("[Trainset size:]",len(train_dataset))
    model = get_model(args)

    training_arguments = TrainingArguments(output_dir=args.output_dir,
                                           overwrite_output_dir=False,
                                           evaluation_strategy='steps',
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
                                           logging_first_step=False,
                                           logging_strategy='steps',
                                           logging_steps=args.logging_steps,
                                           save_strategy='steps',
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
                                           remove_unused_columns=False,
                                           ignore_data_skip=False,
                                           prediction_loss_only=False)
    set_seed(args)
    compute_metric = None
    if args.task in ['yelp_sentiment']:
        compute_metric = compute_metric_sst
    if args.task in ["yelp_food_type3"]:
        compute_metric = compute_metric
    trainer = Trainer(model=model,
                      args=training_arguments,
                      train_dataset=train_dataset,
                      eval_dataset=dev_dataset,
                      compute_metrics=compute_metric)

    if args.mode.endswith('train'):
        if args.local_rank in [-1, 0]:
            print('***** training arguments *****')
            print('warmup steps: ', args.warmup_steps)
            print('lr: ', args.learning_rate)
            print('fp16: ', args.fp16)
            print('fp16-level: ', args.fp16_opt_level)
            print('task: ', args.task)
            print('model path: ', args.model_path)
        trainer.train(model_path = args.model_path)

    if args.mode.endswith('eval'):
        # predict and evaluate on dev_dataset
        trainer.evaluate()

    if args.mode.endswith('test'):
        trainer.evaluate(eval_dataset=test_dataset)

    if args.mode.endswith('txt'):
        import time
        start_time = time.time()
        # predict and evaluate on test_dataset
        tokenizer = RobertaTokenizerFast.from_pretrained('checkpoints/roberta-large')
        if args.mode == 'eval_txt':
            dataset = create_cla_dataset_from_txt(args.txt, args.task, tokenizer)
            train_dataset, dev_dataset, test_dataset = dataset, dataset, dataset
            res = trainer.predict(test_dataset) 
            m = torch.nn.Softmax(dim=1)
            socres = m(torch.Tensor(res.predictions))
            socres = socres.numpy()
            predictions = np.array(res.predictions)
            predictions_ids = np.argmax(predictions, axis=1)
            total_num = len(predictions_ids)
            label_num = [0.0, 0.0, 0.0, 0.0]
            for i in predictions_ids:
                label_num[i] += 1.0
            label_total = [total_num] * 4
            label_percent = np.array(label_num) / np.array(label_total)
            for i in range(len(label_num)):
                if label_num[i] > 0:
                    print('label {} ---- num :{}, percent: {}'.format(i, label_num[i], label_percent[i]))
            print('total_num: ', total_num)
        if args.mode == 'dir_txt':
            dataset_dirs = os.listdir(args.txt)
            sorted_index = get_sorted_index(args.txt)
            dir_res = []
            for dataset_name in dataset_dirs:
                dataset_dir = os.path.join(args.txt, dataset_name)
                dataset = create_cla_dataset_from_txt(dataset_dir, args.task, tokenizer)
                train_dataset, dev_dataset, test_dataset = dataset, dataset, dataset
                res = trainer.predict(test_dataset)
                predictions = np.array(res.predictions)
                predictions_ids = np.argmax(predictions, axis=1)
                total_num = len(predictions_ids)
                label_num = [0.0, 0.0, 0.0, 0.0]
                for i in predictions_ids:
                    label_num[i] += 1.0
                label_total = [total_num] * 4
                label_percent = np.around(np.array(label_num) / np.array(label_total), 4)
                dir_res.append({'txt': dataset_name, 'label_num': label_num, 'total_num': total_num,
                                'percent': label_percent})
            dir_res.sort(key=lambda x: x['percent'][sorted_index])
            print('')
            for curr_res in dir_res:
                print('{} : {}'.format(curr_res['txt'], curr_res['percent']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1, help='local_rank for distributed training on gpus')
    parser.add_argument('--task', type=str, choices=["yelp_sentiment","yelp_food_type3"],
                        help='the task for training and evaluating')
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'test', 'eval_txt', 'dir_txt'], default='train',
                        help='train,eval,test using dataset, eval_txt evaluating the sentences in txt')
    parser.add_argument('--txt', type=str, default=None, help='path of txt file to be evaluated')
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
    parser.add_argument('--fp16', action='store_true', help='whether to use fp16')
    parser.add_argument('--fp16_opt_level', choices=['O0', 'O1', 'O2', 'O3'], default='O0',
                        help='For fp16: Apex AMP optimization level selected in ["O0", "O1", "O2", and "O3"]')
    parser.add_argument('--no_cuda', action='store_true', help='training on cpu if set')
    parser.add_argument('--seed', type=int, default=42, help='random seed for initialization')
    parser.add_argument('--per_device_train_batch_size', default=16, type=int,
                        help='batch size per GPU/CPU for training')
    parser.add_argument('--per_device_eval_batch_size', default=32, type=int,
                        help='batch size per GPU/CPU for evaluating')
    parser.add_argument('--model_path', type=str, default=None,
                        help='checkpoint path to load for continuing-training or evaluate')
    parser.add_argument('--logging_dir', default='./logs', type=str, help='dir to save logs')
    parser.add_argument('--run_name', type=str, default='run_1', help='trainer run name')
    parser.add_argument('--datasize', type=float, default=1.0,
                        help='how much data used in training, e.g. 0.5 means half.')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='whether or not to use gradient checkpointing')
    parser.add_argument('--save_dir', type=str, default='', help='dir to save preidctions')
    args = parser.parse_args()
    if args.mode=="eval_txt":
        print("[Evaluating the file from {}]".format(args.txt))
    main(args)
