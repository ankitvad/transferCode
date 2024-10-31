import torch
import numpy as np
import nltk
import pandas as pd
import pickle as p
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, AutoTokenizer, TrainingArguments, pipeline, logging, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Dataset
from evaluate import load
import sys
from tqdm import tqdm

#CUDA_VISIBLE_DEVICES=0 python bart.py /datasets/ankitUW/resources/grndtr_data/EN/train.csv /datasets/ankitUW/resources/grndtr_data/EN/dev.csv l2-0-I/

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train = sys.argv[1]#training file
dev = sys.argv[2]#dev file
op_dir = sys.argv[3]

def loadData(f):
	df = pd.DataFrame()
	x = open(f,"r").read().strip().split("\n")# s, t
	src = []
	trg = []
	for j in x:
		tmp = j.split("|\t|")
		assert(len(tmp) == 2)
		src.append(tmp[0])
		trg.append(tmp[1])
	df["src"] = src
	df["trg"] = trg
	return df


train_ds = loadData(train)
val_ds = loadData(dev)
train_dataset = Dataset.from_pandas(train_ds)
val_dataset = Dataset.from_pandas(val_ds)

model_name = "/datasets/model/bart-location"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"

def preprocess_function(examples):
	model_inputs = tokenizer(examples['src'], max_length=72, truncation=True)
	labels = tokenizer(text_target=examples['trg'], max_length=72, truncation=True)
	model_inputs['labels'] = labels['input_ids']
	return model_inputs

train_ds_tok = train_dataset.map(preprocess_function, batched=True)
val_ds_tok = val_dataset.map(preprocess_function, batched=True)

metric_wer = load('wer')

def compute_metrics(eval_preds):
	preds, labels = eval_preds
	# decode preds and labels
	labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
	decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
	decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
	# rougeLSum expects newline after each sentence
	decoded_preds = [pred.strip() for pred in decoded_preds]
	decoded_labels = [label.strip() for label in decoded_labels]
	result = metric_wer.compute(predictions=decoded_preds, references=decoded_labels)
	return result


"""## Model"""
model = AutoModelForSeq2SeqLM.from_pretrained('model_name')
# Batching function
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

op_dir = "/datasets/ankitUW/bart-ft/" + op_dir

# Define arguments of the finetuning
training_args = Seq2SeqTrainingArguments(
	output_dir=op_dir,
	evaluation_strategy='epoch',
	learning_rate=3e-5,
	per_device_train_batch_size=32,# batch size for train
	per_device_eval_batch_size=16,
	gradient_accumulation_steps=1,
	eval_strategy="epoch",
	save_strategy="epoch",
	save_total_limit=2,# num of checkpoints to save
	load_best_model_at_end = True,
	num_train_epochs=10,
	fp16=True,
	predict_with_generate=True,
	dataloader_num_workers=10,
	greater_is_better=False,
	group_by_length=True,
	report_to="none"
)

trainer = Seq2SeqTrainer(
	model=model,
	args=training_args,
	train_dataset=train_ds_tok,
	eval_dataset=val_ds_tok,
	tokenizer=tokenizer,
	data_collator=data_collator,
	compute_metrics=compute_metrics
)

trainer.train()


pipe = pipeline(task="text2text-generation", model = model, tokenizer = tokenizer, max_new_tokens = 72, batch_size=32)

dev_sents = open(dev,"r").read().strip().split("\n")
s_t_h = {"s":[], "t":[], "h":[]}

for d in tqdm(dev_sents):
	tmp = d.split("|\t|")
	assert(len(tmp) == 2)
	s = tmp[0]
	t = tmp[1]
	result = pipe(s)
	h = result[0]['generated_text']
	s_t_h["s"].append(s)
	s_t_h["t"].append(t)
	s_t_h["h"].append(h)

p.dump(s_t_h, open(op_dir+"s_t_h_infer.p", "wb"))