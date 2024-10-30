#CUDA_VISIBLE_DEVICES=0 python inferResults.py /datasets/ankitUW/llama-ft/l3-0-I/

import sys

fLoc = sys.argv[1]
op_dir = fLoc[:]
fLoc = fLoc.strip("/").split("/")[-1]#l2-0-I
fLoc = fLoc.split("-")
assert(len(fLoc) == 3)

mType = fLoc[0]

if fLoc[0] == "l2":
	base_model_path = "/datasets/model/llama2-7b-chat"
elif fLoc[0] == "l3":
	base_model_path = "/datasets/model/Meta-Llama-3-8B-Instruct"

topk = int(fLoc[1])

if fLoc[2] == "I":
	train_file = "/datasets/ankitUW/resources/grndtr_data/EN/train.csv"
elif fLoc[2] == "NI":
	train_file = "/datasets/ankitUW/resources/grndtr_data/EN/train_proc.csv"

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever as BM25
import pandas as pd

lDoc = []
x = open(train_file,"r").read().strip().split("\n")# s, t
for i in x:
	tmp = i.split("|\t|")
	s = tmp[0]
	t = tmp[1]
	tmp = Document(page_content = s, metadata = dict(aws = s, man = t))
	lDoc.append(tmp)
retbm25 = BM25.from_documents(lDoc)
retbm25.k = topk


import sys, gc, os
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig, pipeline, logging
from trl import SFTTrainer
from transformers import StoppingCriteria, StoppingCriteriaList
import pickle as p
from tqdm import tqdm
import jiwer
import numpy as np

logging.set_verbosity(logging.CRITICAL)

#Load Models Etc:

# bitsandbytes parameters
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False

compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
	load_in_4bit=use_4bit,
	bnb_4bit_quant_type=bnb_4bit_quant_type,
	bnb_4bit_compute_dtype=compute_dtype,
	bnb_4bit_use_double_quant=use_nested_quant,
)

device_map = {"": 0}

base_model = AutoModelForCausalLM.from_pretrained(
	base_model_path,
	quantization_config=bnb_config,
	device_map=device_map
)
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1
new_model = op_dir+"final-model"
model_f = PeftModel.from_pretrained(base_model, new_model)


stop_token_list = ["</sent>"]
stop_token_ids = [tokenizer(x, add_special_tokens=False)['input_ids'] for x in stop_token_list]

class StopOnTokens(StoppingCriteria):
	def __call__(self, input_ids:torch.LongTensor, scores:torch.FloatTensor, **kwargs) -> bool:
		for stop_ids in stop_token_ids:
			last_ids = input_ids[:,-len(stop_ids):].tolist()
			return stop_ids in last_ids

stop_critr = StoppingCriteriaList([StopOnTokens()])
pipe = pipeline(task="text-generation", model = model_f, tokenizer = tokenizer, max_new_tokens = 100, stopping_criteria=stop_critr)

dev_file = "/datasets/ankitUW/resources/grndtr_data/EN/dev.csv"
dev_sents = open(dev_file,"r").read().strip().split("\n")
s_t_h = {"s":[], "t":[], "h":[]}


for d in tqdm(dev_sents):
	tmp = d.split("|\t|")
	assert(len(tmp) == 2)
	s = tmp[0]
	t = tmp[1]
	retdocs = retbm25.invoke(s)
	examples = []
	for i in retdocs:
		examples.append({"aws": i.metadata["aws"], "man": i.metadata["man"]})
	#L2:
	prefixL2 = "[INST] <<SYS>>\nYou are an automatic speech recognition transcript error correction tool. You are shown the automatically generated transcripts of calls between a user and a customer service support agent. Check the speech transcript sentence and correct all possible errors.\n<</SYS>>"
	example_instrL2 = "\nHere are some examples of speech transcripts and their corrections :"
	endTagL2 = "\n[/INST]\n"
	#L3:
	prefixL3 = "<|start_header_id|>system<|end_header_id|>\n\nYou are an automatic speech recognition transcript error correction tool. You are shown the automatically generated transcripts of calls between a user and a customer service support agent. Check the speech transcript sentence and correct all possible errors.<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\n"
	example_instrL3 = "Here are some examples of speech transcripts and their corrections :<|eot_id|>\n"
	if mType == "l2":
		cStr = prefixL2 + example_instrL2 + endTagL2
	elif mType == "l3":
		cStr = prefixL3 + example_instrL3
	else:
		return 0
	for e in examples:
		cStr += "<sent> Speech Transcript: " + e["aws"] +"\nCorrection: " + e["man"] + "</sent>\n"
	cStr += "<sent> Speech Transcript: " + s +"\nCorrection: "
	result = pipe(cStr)
	h = result[0]['generated_text']
	s_t_h["s"].append(s)
	s_t_h["t"].append(t)
	s_t_h["h"].append(h)

p.dump(s_t_h, open(op_dir+"few_shot_res_final.p", "wb"))