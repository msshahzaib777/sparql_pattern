import datasets
import pickle
import numpy as np
import pandas as pd
import random
import torch
print("Clearing Cache")
torch.cuda.empty_cache()
import gc
gc.collect()
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
device
print("setting config")
MAX_LENGTH = 1024
BATCH_SIZE = 2
EOS = "<|endoftext|>"
PAD = "<|pad|>"
SOS = "<|startoftext|>"

print("loading Data")
with open('SPARQL_PTRN_50k.pickle', 'rb') as f:
    dataset = pickle.load(f)

print("loading model")
from transformers import GPT2Tokenizer, GPT2LMHeadModel
checkpoint = "sparql_model_gpt2_2/checkpoint-966"
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

print("processing Data")
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

print("Setting Trainer Arg")
from transformers import TrainingArguments, Trainer, Seq2SeqTrainingArguments, TrainerCallback
model.generation_config.use_cache = False 
training_args = Seq2SeqTrainingArguments(
    output_dir="sparql_model_gpt2_2",
    evaluation_strategy="steps",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    num_train_epochs=2,
    gradient_accumulation_steps = 3,
    save_total_limit= 1,
    load_best_model_at_end= True,
    predict_with_generate=True,
    fp16=True,
    logging_steps= 700,
    logging_dir='./logs',
    eval_steps= 700,
    save_steps= 700
)

class PrinterCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            print(logs)

    def on_step_begin(self, args, state, control, logs=None, **kwargs):
        print("Step end", logs)
    def on_step_end(self, args, state, control, logs=None, **kwargs):
        print("Step begin", logs)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    callbacks=[PrintCallback]
)

print("Training ... ")
trainer.train()

# from tqdm.notebook import tqdm
# correct = 0
# generated = []
# for i in tqdm(range(0, 1000)):
#     sample = tokenized_dataset["test"][i]
    
#     sample_idx = sample['sum_idx']-1
#     response = model.generate(torch.tensor([sample["input_ids"][:sample_idx]]).cuda(), \
#                               attention_mask = torch.tensor([sample["attention_mask"][:sample_idx]]).cuda(),
#                                max_length=len(sample["input_ids"])+5, temperature=1.0,
#                              top_k=50,
#                              top_p=0.95,
#                              repetition_penalty=1.0,
#                              do_sample=True,
#                              num_return_sequences=1,
#                              length_penalty=2.0,
#                              early_stopping=True, pad_token_id=tokenizer.pad_token_id, use_cache=False)
#     predicted_query = tokenizer.decode(response[0][sample["sum_idx"]-1:-1]).strip()
#     actual_query = sample["sparql"]
#     generated.append({"sample": sample, 'predicted query': predicted_query})
#     if(predicted_query == actual_query.replace("?", "") ):
#         correct +=1
