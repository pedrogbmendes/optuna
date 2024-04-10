import optuna

#pip3 install optuna optuna-dashboard
#optuna-dashboard sqlite:///study.db
#/Users/pedromendes/Library/Python/3.8/bin/optuna-dashboard sqlite:///study.db
# http://127.0.0.1:8080/dashboard/

#pip3 install gpytorch GPy evaluate datasets mpmath rouge_score

import torch
import torch.nn as nn
from torch.optim import AdamW
#import torch.nn.functional as F
#from torch.utils.data import DataLoader
#import torch.multiprocessing as multiprocessing
import multiprocessing

import logging, sys

#from transformers import BartTokenizer, BartForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration, BartModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
from transformers import Adafactor
import datasets,  evaluate #, nltk
import argparse, os

import numpy as np

#import GPUtil
#import gc

torch.autograd.set_detect_anomaly(True)


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
MAX_LENGHT = 1024 #256 #512
GPU_BATCH_SIZE = 2

DEBUG = True #False
test_mode = True

#sumarization tasks
# https://huggingface.co/datasets/EdinburghNLP/xsum
# https://huggingface.co/datasets/cnn_dailymail
# https://huggingface.co/docs/transformers/model_doc/bart
# https://huggingface.co/docs/transformers/model_doc/t5



class dataset():
    def __init__(self, model_name, dataset_name):
        # Initialize the BART model and tokenizer
        self.model_name = model_name
        self.dataset_name = dataset_name

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, model_max_length=MAX_LENGHT)

        # Load the CNN/DailyMail dataset
        if 'cnn' in self.dataset_name or 'dailymail' in self.dataset_name:
            raw_datasets = datasets.load_dataset("cnn_dailymail", "3.0.0")
            mapFucntion = self.preprocess_dailymail

        elif 'xsum' in self.dataset_name:
            raw_datasets = datasets.load_dataset("xsum") #, trust_remote_code=True)
            mapFucntion = self.preprocess_xsum
               
        else:
            print("No implementation for this dataset!!!")
            sys.exit(0)

        dataset_train = raw_datasets['train']
        dataset_test = raw_datasets['validation']

        if DEBUG:
            subset_size = 10
            dataset_train = dataset_train.shuffle(seed=42).select([i for i in range(subset_size)])

        self.data_train = dataset_train.map(mapFucntion, batched=True,)
        self.data_train.set_format('torch', columns=['input_ids', 'labels'])

        if DEBUG:
            subset_size = 10
            dataset_test = dataset_test.shuffle(seed=42).select([i for i in range(subset_size)])
        
        self.data_eval = dataset_test.map(mapFucntion, batched=True,)
        self.data_eval.set_format('torch', columns=['input_ids', 'labels'])


    def preprocess_dailymail(self, example):
        #inputs = self.tokenizer(example["article"], return_tensors="pt", max_length=self.max_length, truncation=True)
        inputs = self.tokenizer(example["article"], return_tensors="pt", padding=True, truncation=True)
        #targets = self.tokenizer(example["highlights"], return_tensors="pt", max_length=self.max_length, truncation=True)
        targets = self.tokenizer(example["highlights"], return_tensors="pt", padding=True, truncation=True)
        inputs["labels"] = targets["input_ids"]
        return inputs


    def preprocess_xsum(self, example):
        #inputs = self.tokenizer(example["document"], return_tensors="pt", max_length=self.max_length, truncation=True)
        inputs = self.tokenizer(example["document"], return_tensors="pt", padding=True, truncation=True)
        #targets = self.tokenizer(example["summary"], return_tensors="pt", max_length=self.max_length, truncation=True)
        targets = self.tokenizer(example["summary"], return_tensors="pt", padding=True, truncation=True)
        inputs["labels"] = targets["input_ids"]
        return inputs
    

class Seq2SeqTrainer_():
#class Seq2SeqTrainer_(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")

        outputs = model(**inputs)
        logits = outputs.get("logits")
        #generated_ids = logits.argmax(dim=-1)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.config.vocab_size), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


class ModelSystem():
    def __init__(self, model_name, dataset_name, data_train, data_eval):

        # Initialize the BART model and tokenizer
        self.model_name = model_name
        self.model_name_write = self.model_name.replace('/', '-')
        self.dataset_name = dataset_name

        self.prints = True
        self.data_eval = data_eval
        self.data_train = data_train
        self.debug = DEBUG
            

    def load_model(self, budget, allbudget, model_name_save, optimizer_type, learning_rate, weight_decay, scheduler_lr, num_training_steps, num_warmup_steps):

        if allbudget is None:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, device_map=None,).to(DEVICE)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, model_max_length=MAX_LENGHT)

            self.optimizer = None
            self.scheduler = None

            return 0.0


        else:
            for budget_inter in reversed(allbudget):
                if budget_inter >= budget: continue

                if self.debug:
                    name = './checkpoints/' + self.model_name_write + '_' + self.dataset_name + '_' + model_name_save + '_budget' + str(budget_inter) + '/'
                else:
                    name = '../nas/checkpoints/' + self.model_name_write + '_' + self.dataset_name + '_' + model_name_save + '_budget' + str(budget_inter) + '/'
                if os.path.isdir(name):
                    print("Loading model, tokenizer, optimizer, and scheduler")
                    try:
                        self.model = AutoModelForSeq2SeqLM.from_pretrained(name, device_map=None,).to(DEVICE)
                        self.tokenizer = AutoTokenizer.from_pretrained(name)
                    except:
                        print("Error Loading the model and tokenizer!")
                        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, device_map=None,) #, model_max_length=MAX_LENGHT)
                        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(DEVICE)
                        self.scheduler = None
                        self.optimizer = None
                        return 0.0
                    
                    checkpoint = torch.load(name + 'optimizer_scheduler.pt', map_location=DEVICE)

                    if optimizer_type == "Adafactor":
                        self.optimizer = Adafactor(self.model.parameters(),
                            lr=learning_rate,
                            weight_decay=weight_decay, 
                            relative_step=False)
                    else:
                        self.optimizer = AdamW(self.model.parameters(),
                            lr=learning_rate,
                            weight_decay=weight_decay)

                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


                    if scheduler_lr == "linear":
                        scheduler_function = get_linear_schedule_with_warmup
                    elif scheduler_lr == "cosine":
                        scheduler_function = get_cosine_schedule_with_warmup
                    else:
                        scheduler_function = get_polynomial_decay_schedule_with_warmup


                    self.scheduler = scheduler_function(
                                    optimizer=self.optimizer,
                                    num_warmup_steps=num_warmup_steps,  # Warm-up proportion can be adjusted
                                    num_training_steps=num_training_steps
                                ) 
                    #try:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    #except:
                    #    print("Error Loading the scheduler and optimizer!")
                    #    self.scheduler = None
                    #    self.optimizer = None

                    return budget_inter

        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, device_map=None).to(DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name) #, model_max_length=MAX_LENGHT)
        self.optimizer = None
        self.scheduler = None
        return 0.0


    def update_hyperparameter(self, num_epochs=1, learning_rate=1e-5, batch_size=4, weight_decay=1e-4, max_length=512,scheduler_lr="linear", num_warmup_steps=10, \
                              num_training_steps=10, optim='AdamW', allbudget=None, hp_name=""):


        last_available_budget = self.load_model(num_epochs, allbudget, hp_name, optim, \
                                                          learning_rate, weight_decay, scheduler_lr, num_training_steps, num_warmup_steps)


        training_args = Seq2SeqTrainingArguments(
            evaluation_strategy="no",
            save_strategy='no',
            save_total_limit=0,
            #save_steps=500,
            #save_steps=10,
            #eval_steps=10,
            #include_inputs_for_metrics=True,
            
            per_device_train_batch_size=GPU_BATCH_SIZE, #self.batch_size,
            per_device_eval_batch_size=GPU_BATCH_SIZE,
            eval_accumulation_steps = 40,
            predict_with_generate=True,  # Whether to use generate to calculate generative metrics (ROUGE, BLEU)
            generation_num_beams=5,

            output_dir="./checkpoints",

            gradient_accumulation_steps=batch_size,
            #gradient_checkpointing=True,

            remove_unused_columns=False,
            num_train_epochs=num_epochs-last_available_budget,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            push_to_hub=False,
           
            #fp16=True,  # Enable mixed precision
            #fp16_full_eval=True
        )


        if self.optimizer is None:
            if optim == "Adafactor":
                self.optimizer = Adafactor(self.model.parameters(),
                                lr=learning_rate,
                                weight_decay=weight_decay, 
                                relative_step=False)
            else:
                self.optimizer = AdamW(self.model.parameters(),
                                lr=learning_rate,
                                weight_decay=weight_decay)
        
        if self.scheduler is None:
            if scheduler_lr == "linear":
                scheduler_function = get_linear_schedule_with_warmup
            elif scheduler_lr == "cosine":
                scheduler_function = get_cosine_schedule_with_warmup
            else:
                scheduler_function = get_polynomial_decay_schedule_with_warmup


            self.scheduler = scheduler_function(
                    optimizer=self.optimizer,
                    num_warmup_steps=num_warmup_steps,  # Warm-up proportion can be adjusted
                    num_training_steps=num_training_steps
                ) 

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=False,
            label_pad_token_id=self.tokenizer.pad_token_id,
        )


        # Fine-tune the model
        self.trainer = Seq2SeqTrainer_(
            #model_name=self.model_name,
            model=self.model,
            tokenizer=self.tokenizer,
            args=training_args,
            data_collator=data_collator,
            train_dataset=self.data_train, #dataset["train"],
            eval_dataset=self.data_eval, #dataset["validation"],
            compute_metrics=self.compute_metrics,
            optimizers=(self.optimizer, self.scheduler),
        )


    def train(self,):
        self.model.train()
        self.trainer.train()


    def save(self, model_name_save, budget):
        # Save the fine-tuned model
        if self.debug:
            name = './checkpoints/' + self.model_name_write + '_' + self.dataset_name + '_' + model_name_save + '_budget' + str(budget) + '/'
        else:
            name = '../nas/checkpoints/' + self.model_name_write + '_' + self.dataset_name + '_' + model_name_save + '_budget' + str(budget) + '/'
        self.model.save_pretrained(name)
        self.tokenizer.save_pretrained(name)

        torch.save({
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, name + 'optimizer_scheduler.pt')


    @torch.no_grad()
    def compute_metrics(self, eval_preds):
        
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        #preds = np.argmax(preds, axis=-1)
        
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        rouge = evaluate.load('rouge')
        #result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
        return result


    def test(self,):
        self.model.eval()
        rouge =  self.trainer.evaluate()
        if 'rouge2' in rouge.keys():
            rouge2 = rouge['rouge2']
        elif 'eval_rouge2' in rouge.keys():
            rouge2 = rouge['eval_rouge2']
        else:
            rouge2 = rouge.values()[0]
        print(rouge)
        return rouge2


# def print_gpu_memory():
#     # Get the list of available GPUs
#     gpus = GPUtil.getGPUs()

#     for i, gpu in enumerate(gpus):
#         print()
#         print(f"GPU {i + 1}:")
#         print(f"  ID: {gpu.id}")
#         print(f"  Name: {gpu.name}")
#         print(f"  Memory Total: {gpu.memoryTotal} MB")
#         print(f"  Memory Used: {gpu.memoryUsed} MB")
#         print(f"  Memory Free: {gpu.memoryFree} MB")
#         print(f"  Memory Percentage: {gpu.memoryUtil * 100}%")
#         print()
#         print()


# def print_gpu_memory1():
#     # Get the list of available GPUs
#     gpus = GPUtil.getGPUs()

#     for i, gpu in enumerate(gpus):
#         print(f"GPU  ID: {gpu.id}")
#         print(f"  Memory Percentage: {gpu.memoryUtil * 100}%")
#         print()
#         break


def deploy_model_hp(data_train, data_eval, model_name, dataset_name, budget, learning_rate, batch_size, weight_decay, \
                                scheduler_lr, scheduler_lr_steps, scheduler_lr_warmup, optimizer, allbudget, hp_name, queue):

    model_trainer =  ModelSystem(model_name, dataset_name, data_train, data_eval)

    model_trainer.update_hyperparameter(num_epochs=budget, 
                    learning_rate=learning_rate, 
                    batch_size=batch_size, 
                    weight_decay=weight_decay, 
                    max_length=MAX_LENGHT,
                    scheduler_lr= scheduler_lr,
                    num_warmup_steps=scheduler_lr_steps, 
                    num_training_steps=scheduler_lr_warmup, 
                    optim=optimizer,
                    allbudget=allbudget, 
                    hp_name=hp_name)
    

    model_trainer.train()
    model_trainer.save(hp_name, budget)
    rouge2 = model_trainer.test()
    queue.put(rouge2)

    #del model_trainer.model, model_trainer.tokenizer, model_trainer.optimizer, model_trainer.scheduler, model_trainer.trainer
    #gc.collect()
    #torch.cuda.empty_cache()


def deploy_model_hp_datasetSize(data_train, data_eval, model_name, dataset_name, budget, learning_rate, batch_size, weight_decay, \
                                scheduler_lr, scheduler_lr_steps, scheduler_lr_warmup, optimizer, allbudget, hp_name, queue):
    #budget is dataset size
    subset_dataset = data_train.shuffle(seed=42).select(range(int(len(data_train)*budget)))
    model_trainer =  ModelSystem(model_name, dataset_name, subset_dataset, data_eval)

    model_trainer.update_hyperparameter(num_epochs=2, 
                    learning_rate=learning_rate, 
                    batch_size=batch_size, 
                    weight_decay=weight_decay, 
                    max_length=MAX_LENGHT,
                    scheduler_lr= scheduler_lr,
                    num_warmup_steps=scheduler_lr_steps, 
                    num_training_steps=scheduler_lr_warmup, 
                    optim=optimizer,
                    allbudget=allbudget, 
                    hp_name=hp_name)
    

    model_trainer.train()
    model_trainer.save(hp_name, budget)
    rouge2 = model_trainer.test()
    queue.put(rouge2)

    #del model_trainer.model, model_trainer.tokenizer, model_trainer.optimizer, model_trainer.scheduler, model_trainer.trainer
    #gc.collect()
    #torch.cuda.empty_cache()


def objective(trial):

    if test_mode:
        batch_size = trial.suggest_int("gradient_accumulation_steps", 4, 64) #diminuir para 4
        learning_rate = trial.suggest_float("lr", 1e-7, 1e-5)
        weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-5)
        optimizer = trial.suggest_categorical("optimizer", ['AdamW', 'Adafactor']) 
        scheduler_lr_warmup = trial.suggest_int("num_warmup_steps", 0, 500) # aumento para 500
        scheduler_lr = trial.suggest_categorical("scheduler_lr", ["linear", "poly", "cosine"]) 
        budget = trial.suggest_float(trial.study.sampler.budgetName, trial.study.sampler.min_budget, trial.study.sampler.max_budget)

        rouge2 =  np.random.rand()
        print(rouge2)
        return rouge2

    #batch_size = trial.suggest_int("batch_size", GPU_BATCH_SIZE, 64)
    batch_size = trial.suggest_int("gradient_accumulation_steps", 4, 64) #diminuir para 4

    learning_rate = trial.suggest_float("lr", 1e-7, 1e-5)
    weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-5)

    optimizer = trial.suggest_categorical("optimizer", ['AdamW', 'Adafactor']) 

    scheduler_lr = trial.suggest_categorical("scheduler_lr", ["linear", "poly", "cosine"]) 
    #scheduler_lr_steps = trial.suggest_int("num_training_steps", 10, 1e4)

    #dataset_size = data.train_dataset_size = len(data.data_train)
    dataset_size=60000
    bb = GPU_BATCH_SIZE * batch_size # batch_size=gradient_accumulation_steps

    if hasattr(trial.study.sampler, 'max_budget'):
        scheduler_lr_steps = dataset_size / bb * trial.study.sampler.max_budget
    else:
        scheduler_lr_steps = dataset_size / bb * args.max_budget

    scheduler_lr_warmup = trial.suggest_int("num_warmup_steps", 0, 500) # aumento para 500

    if hasattr(trial.study.sampler, 'budgetName'):
        budget = trial.suggest_float(trial.study.sampler.budgetName, trial.study.sampler.min_budget, trial.study.sampler.max_budget)

        # this is used for model checkpointing 
        allbudget = sampler.getBudgets()
        hp_name = 'batch' + str(batch_size) + '_lr' + str(learning_rate) + \
                                '_weightDecay' + str(weight_decay) + '_opt' + str(optimizer) + \
                                '_schedulerLr' + str(scheduler_lr) + '_schedulerLrSteps' + str(scheduler_lr_steps) +\
                                '_schedulerLrWarmup' + str(scheduler_lr_warmup)
        budgetName = trial.study.sampler.budgetName
    else:
        budget = args.max_budget #trial.suggest_float('epochs', 1, 2)
        allbudget = None
        hp_name = ""
        budgetName = args.budget


    

    result_queue = multiprocessing.Queue()
    if budgetName == 'epochs':
        my_process = multiprocessing.Process(target=deploy_model_hp,args=(data.data_train, data.data_eval, \
                                                                        model_name, dataset_name, \
                                                                        budget, learning_rate, batch_size, weight_decay, \
                                                                        scheduler_lr, scheduler_lr_steps, scheduler_lr_warmup, \
                                                                        optimizer, allbudget, hp_name,result_queue,))
        
    elif budgetName == 'datasetsize':
        my_process = multiprocessing.Process(target=deploy_model_hp_datasetSize,args=(data.data_train, data.data_eval, \
                                                                        model_name, dataset_name, \
                                                                        budget, learning_rate, batch_size, weight_decay, \
                                                                        scheduler_lr, scheduler_lr_steps, scheduler_lr_warmup, \
                                                                        optimizer, allbudget, hp_name,result_queue,))
    else:
        my_process = multiprocessing.Process(target=deploy_model_hp,args=(data.data_train, data.data_eval, \
                                                                        model_name, dataset_name, \
                                                                        budget, learning_rate, batch_size, weight_decay, \
                                                                        scheduler_lr, scheduler_lr_steps, scheduler_lr_warmup, \
                                                                        optimizer, allbudget, hp_name,result_queue,))
        
    my_process.start()
    my_process.join()
    rouge2 = result_queue.get()
    print(rouge2)
    return rouge2


def test_objective():
    #batch_size = trial.suggest_int("batch_size", GPU_BATCH_SIZE, 64)
    batch_size = 1 #trial.suggest_int("gradient_accumulation_steps", 1, 10)
    lr = 1e-5 #trial.suggest_float("lr", 1e-7, 1e-5)
    weight_decay = 1e-5 #trial.suggest_float("weight_decay", 1e-7, 1e-5)


    optimizer = 'AdamW' #trial.suggest_categorical("optimizer", ['AdamW', 'Adafactor']) 

    scheduler_lr = "linear" #trial.suggest_categorical("scheduler_lr", ["linear", "poly", "cosine"]) 
    scheduler_lr_steps = 100 #trial.suggest_int("num_training_steps", 10, 1e4)
    scheduler_lr_warmup = 10 #trial.suggest_int("num_warmup_steps", 0, 100) 

    #if hasattr(trial.study.sampler, 'budgetName'):
    #    budget = trial.suggest_float(trial.study.sampler.budgetName, trial.study.sampler.min_budget, trial.study.sampler.max_budget)
    #else:
    budget = 0.05 #trial.suggest_float('epochs', 1, 2)


    model_trainer =  ModelSystem(num_epochs=budget, 
                    lr=lr, 
                    batch_size=batch_size, 
                    weight_decay=weight_decay, 
                    max_length=MAX_LENGHT,
                    scheduler_lr= scheduler_lr,
                    num_warmup_steps=scheduler_lr_steps, 
                    num_training_steps=scheduler_lr_warmup, 
                    optim=optimizer,
                  )


    model_trainer.model.to(DEVICE)
    #try:
    model_trainer.train()
    #model_trainer.save(budget)
    rouge2 = model_trainer.test()

    del model_trainer.model
    torch.cuda.empty_cache()
    #print_gpu_memory()

    return rouge2


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()

    parser.add_argument('--sampler', type=str, help='sampler', default="hb" )
    parser.add_argument('--n_trials', type=int, help='n_trials', default=40)
    parser.add_argument('--min_budget', type=float, help='min_budget', default=1)
    parser.add_argument('--max_budget', type=float, help='max_budget', default=81.0)
    parser.add_argument('--eta', type=int, help='eta', default=3)


    parser.add_argument('--model', type=str, help='model', default='t5-base') # "facebook/bart-large"
    parser.add_argument('--dataset', type=str, help='dataset', default='xsum') # "cnn_dailymail"

    parser.add_argument('--budget', type=str, help='budget', default='epochs')
    parser.add_argument('--seed', type=int, help='seed', default=10000)

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    print("budget is " + args.budget)

    model_name = args.model
    dataset_name = args.dataset
    
    if not test_mode:
        data = dataset(model_name, dataset_name)

    model_name_write = model_name.replace('/', '-')

    #test_objective()

    # Specify the storage using JournalFileStorage
    #storage_name = 'sqlite:///example.db'  # SQLite is used as an example, you can choose another backend

    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    #study_name = "study_test1"  # Unique identifier of the study.
    study_name = model_name_write + '_' + dataset_name + '_' + args.sampler

    storage = optuna.storages.RDBStorage(
        url="sqlite:///{}.db".format(study_name),
        #heartbeat_interval=60,
        #grace_period=120,
        #failed_trial_callback=optuna.storages.RetryFailedTrialCallback(max_retry=10),
    )

    #storage = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage("./" + study_name + ".log"),)


    if args.sampler=='hb':
        print("Using Hyperband!")
        sampler = optuna.samplers.HyperBand(eta=args.eta, min_budget=args.min_budget,max_budget=args.max_budget, budgetName=args.budget, seed=42)
        #sampler = optuna.samplers.HyperBand(eta= 3, min_budget=1.0,max_budget=81.0,seed=42)
    elif args.sampler=='bohb':
        print("Using BOHB!")
        sampler = optuna.samplers.BOHB(eta=args.eta, min_budget=args.min_budget,max_budget=args.max_budget, budgetName=args.budget, seed=42)
        #sampler = optuna.samplers.HyperBand(eta= 3, min_budget=1.0,max_budget=81.0,seed=42)
    elif args.sampler=='hj':
        print("Using HyperJump!")
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()-1)
        sampler = optuna.samplers.HyperJump(eta=args.eta, min_budget=args.min_budget,max_budget=args.max_budget, budgetName=args.budget, threshold=0.1, seed=42, pool=pool)
        #sampler = optuna.samplers.HyperBand(eta= 3, min_budget=1.0,max_budget=81.0,seed=42)
    else:
        print("Using Random!")
        sampler = optuna.samplers.RandomSampler()

    study = optuna.create_study(direction="maximize",
    #study = optuna.create_study(direction="minimize",
                                sampler=sampler,
                                pruner=None,
                                storage=storage, # 
                                load_if_exists=True,
                                study_name='summarize')
    study.optimize(objective, n_trials=args.n_trials)

    print()
    best_params = study.best_params
    print(best_params)
    print('Best f ' + str(study.best_value))
    print('Best trial ' + str(study.best_trial))

#To RUN
#python3 hpt_summarize.py --sampler=bohb --n_trials=32 --min_budget=0.0625 --max_budget=1.0 --eta=2


#study = optuna.create_study(direction="maximize",sampler=optuna.samplers.RandomSampler(),pruner=None,storage=optuna.storages.JournalStorage(optuna.storages.JournalFileStorage("./journal.log"),),load_if_exists=True,study_name='summarize')
    

# todo:
# Compare models using test set
# evaluate bart-base (ie model without finetuning)
# test other budgets (dataset size, truncation_size, )
    #truncation_size tokenizer(max_length=42)
    #extrapolate results to larger models

    # find workshops around hpt for nlp
    # https://2024.emnlp.org/calls/main_conference_papers/#emnlp-2024-theme-track-efficiency-in-model-algorithms-training-and-inference
    # runs


# NER task
    # https://huggingface.co/dslim/bert-base-NER
    # bert base
    # CoNLL-2003 Named Entity Recognition dataset
    # https://huggingface.co/learn/nlp-course/en/chapter7/2  
    # metric to use f1

# study = optuna.create_study(storage=optuna.storages.RDBStorage(url='sqlite:///facebook-bart-base_dailymail_random.db'), load_if_exists=True,study_name='summarize')
# trials = study.get_trials()
# for trial in trials:
#     if trial.state is not optuna.trial.TrialState.COMPLETE: continue
#     duration = trial.datetime_complete - trial.datetime_start
#     print(trial.values[0])
#     #print(duration.seconds)


