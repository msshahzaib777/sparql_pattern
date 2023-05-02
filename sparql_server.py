import datasets
import pickle
import numpy as np
import pandas as pd
import random
import torch
torch.cuda.empty_cache()
import gc
gc.collect()
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
device

MAX_LENGTH = 1024
BATCH_SIZE = 2
EOS = "<|endoftext|>"
PAD = "<|pad|>"
SOS = "<|startoftext|>"

with open('SPARQL_PTRN_500k.pickle', 'rb') as f:
    dataset = pickle.load(f)


from transformers import GPT2Tokenizer, GPT2LMHeadModel
checkpoint = "sparql_model_gpt2_2/checkpoint-500"
tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
# special_tokens = {'pad_token':'<|pad|>','sep_token':'<|sep|>', 'bos_token': '<|startoftext|>'}
# num_add_toks = tokenizer.add_special_tokens(special_tokens)
model = GPT2LMHeadModel.from_pretrained(checkpoint)
# model.resize_token_embeddings(len(tokenizer))
model.cuda()
model.gradient_checkpointing_enable()
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


def preprocess_function(examples):
        _examples = []
        attention_mask = []
        sum_idx  = []
        for i in range(0, len(examples["question"])):
            context = "Convert this Question into SPARQL query using provided knowledge graph.\n Question: " + examples["question"][i] + \
            "\n knowledge graph: " + examples["rdf"][i] + " " + tokenizer.sep_token
            sparql = examples["sparql"][i].replace("?", "")
            query = "\n SPARQL query: " + sparql + " " + tokenizer.eos_token
            model_inputs = tokenizer(context + query, max_length=MAX_LENGTH, padding='max_length', truncation=True)
            _examples.append(model_inputs["input_ids"])
            attention_mask.append(model_inputs["attention_mask"])
            sum_idx.append(len(tokenizer.encode(context + "\n SPARQL query: ")))
        
        return {"input_ids": _examples, "label": _examples, "attention_mask": attention_mask, "sum_idx": sum_idx}

tokenized_dataset = dataset.map(preprocess_function, batched=True)


# tokenized_evalset = []
# for i in range(0,1500):
#     tokenized_evalset.append(tokenized_dataset["test"][i])
# tokenized_testset = datasets.Dataset.from_pandas(pd.DataFrame(data=tokenized_evalset))    


# tokenized_trainset = []
# for i in range(10000,15000):
#     tokenized_trainset.append(tokenized_dataset["train"][i])
# tokenized_trainset = datasets.Dataset.from_pandas(pd.DataFrame(data=tokenized_trainset))    


from transformers import TrainingArguments, Trainer, Seq2SeqTrainingArguments
model.generation_config.use_cache = False 
training_args = Seq2SeqTrainingArguments(
    output_dir="sparql_model_gpt2_2",
    evaluation_strategy="steps",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    num_train_epochs=5,
    gradient_accumulation_steps = 2,
    save_total_limit= 3,
    load_best_model_at_end= True,
    predict_with_generate=True,
    fp16=True,
    logging_steps= 250,
    eval_steps= 250,
    save_steps= 250
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
)
trainer.train()
