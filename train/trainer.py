import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import csv
import os
from dataclasses import dataclass
from common.utils import set_seed
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
import math

@dataclass
class IDCardConfig:
    model_name_or_path: str
    tokenizer_name_or_path: str
    data_path: str
    valid_path: str  # 新增验证集路径
    output_dir: str
    max_length: int
    batch_size: int
    learning_rate: float
    num_train_epochs: int
    seed: int
    device: str
    fp16: bool
    lora_r: float
    lora_alpha: float
    lora_dropout: float
    accumulation_steps: int
    save_steps: int
    save_full_model: bool
    early_stop_patience: int
    min_delta: float
    eval_steps: int


class IDCardDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_data(file_path)
        
    def load_data(self, file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                question = row[0]
                answer = row[1]
                data.append({
                    'question': question,
                    'answer': answer
                })
        return data
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        answer = item['answer']
        
        prompt = f"问题：{question} 答案："
        text = prompt + answer
    
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        labels = encoding['input_ids'].clone()
    
        prompt_tokens = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        prompt_length = prompt_tokens['input_ids'].shape[1]
        
        labels[:, :prompt_length] = -100
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': labels.flatten()
        }
    
class IDCardTrainer:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name_or_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        kwargs = {
            "max_memory": {i: "24GiB" for i in range(4)},
            "device_map": "auto",
        }
        
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            torch_dtype=torch.bfloat16 if config.fp16 else torch.float32,
            **kwargs
        )

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "v_proj"]
        )

        self.model = get_peft_model(base_model, peft_config)
        self.model.to(self.device)

        self.model.print_trainable_parameters()

        self.train_dataset = IDCardDataset(config.data_path, self.tokenizer, config.max_length)
        self.train_dataloader = DataLoader(
            self.train_dataset, 
            batch_size=config.batch_size, 
            shuffle=True
        )

        self.valid_dataset = IDCardDataset(config.valid_path, self.tokenizer, config.max_length)
        self.valid_dataloader = DataLoader(
            self.valid_dataset,
            batch_size=config.batch_size,
            shuffle=False
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate
        )

        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=len(self.train_dataloader) * config.num_train_epochs
        )

        self.scaler = torch.amp.GradScaler(enabled=config.fp16)

        self.best_valid_loss = float('inf')
        self.epochs_no_improve = 0
        self.early_stop = False
        
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        total_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(self.valid_dataloader, desc="Validating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                with torch.amp.autocast(enabled=self.config.fp16, device_type=self.device):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                    
                total_loss += loss.item()
                total_steps += 1
        
        avg_loss = total_loss / total_steps
        return avg_loss
    
    def train(self):
        os.makedirs(self.config.output_dir, exist_ok=True)
        global_step = 0
        accumulation_steps = self.config.accumulation_steps
        
        for epoch in range(self.config.num_train_epochs):
            if self.early_stop:
                print(f"Early stopping triggered at epoch {epoch}")
                break
                
            print(f"Epoch {epoch+1}/{self.config.num_train_epochs}")
            self.model.train()
            epoch_loss = 0
            total_steps = 0

            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}")
            
            for step, batch in enumerate(progress_bar):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                with torch.amp.autocast(enabled=self.config.fp16, device_type=self.device):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                    
                if not loss.requires_grad:
                    print(f"Warning: Loss does not require grad at step {global_step}")
                    continue
                
                loss = loss / accumulation_steps
                self.scaler.scale(loss).backward()

                if (step + 1) % accumulation_steps == 0 or (step + 1) == len(self.train_dataloader):
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.lr_scheduler.step()
                
                epoch_loss += loss.item() * accumulation_steps
                total_steps += 1
                global_step += 1

                progress_bar.set_postfix({
                    'loss': f"{loss.item() * accumulation_steps:.4f}", 
                    'lr': f"{self.lr_scheduler.get_last_lr()[0]:.8f}"
                })
            
                if global_step % self.config.eval_steps == 0:
                    valid_loss = self.evaluate()
                    print(f"Step {global_step} - Validation Loss: {valid_loss:.4f}")

                    if valid_loss < self.best_valid_loss - self.config.min_delta:
                        self.best_valid_loss = valid_loss
                        self.epochs_no_improve = 0
                        self.save_model("best")
                    else:
                        self.epochs_no_improve += 1
                        print(f"No improvement for {self.epochs_no_improve} evaluations")
                    
                    if self.epochs_no_improve >= self.config.early_stop_patience:
                        self.early_stop = True
                        print("Early stopping criteria met")
                        break
                    
                    self.model.train()

            avg_epoch_loss = epoch_loss / total_steps
            print(f"Epoch {epoch+1} Average Loss: {avg_epoch_loss:.4f}")
            
            valid_loss = self.evaluate()
            print(f"Epoch {epoch+1} - Validation Loss: {valid_loss:.4f}")
            
            if valid_loss < self.best_valid_loss - self.config.min_delta:
                self.best_valid_loss = valid_loss
                self.epochs_no_improve = 0
                self.save_model("best")
            else:
                self.epochs_no_improve += 1
                print(f"No improvement for {self.epochs_no_improve} epochs")

            if self.epochs_no_improve >= self.config.early_stop_patience:
                self.early_stop = True
                print("Early stopping criteria met")

            if (epoch+1) % self.config.save_steps == 0:
                self.save_model(f"epoch_{epoch+1}")
    
    def save_model(self, checkpoint_name):
        output_dir = os.path.join(self.config.output_dir, f"checkpoint_{checkpoint_name}")
        os.makedirs(output_dir, exist_ok=True)

        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        if self.config.save_full_model:
            merged_model = self.model.merge_and_unload()
            merged_model.save_pretrained(os.path.join(output_dir, "full_model"))
        
        print(f"Model saved to {output_dir}")