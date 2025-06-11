import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import csv
import os
from dataclasses import dataclass
from common.utils import set_seed
from peft import LoraConfig, get_peft_model, TaskType

@dataclass
class IDCardConfig:
    model_name_or_path: str
    tokenizer_name_or_path: str
    data_path: str
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
    


class IDCardDataset(Dataset):
    def __init__(self, file_path,tokenizer, max_length=512):
        
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
            torch_dtype=torch.float16 if config.fp16 else torch.float32
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

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate
        )

        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=len(self.train_dataloader) * config.num_train_epochs
        )

        self.scaler = torch.amp.GradScaler(enabled=config.fp16)
        
    def train(self):
        os.makedirs(self.config.output_dir, exist_ok=True)
        self.model.train()
        
        global_step = 0

        accumulation_steps = self.config.gradient_accumulation_steps if hasattr(self.config, 'gradient_accumulation_steps') else 1
        
        for epoch in range(self.config.num_train_epochs):
            print(f"Epoch {epoch+1}/{self.config.num_train_epochs}")
            
            epoch_loss = 0

            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}")
            
            for step, batch in enumerate(progress_bar):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                self.model.train()
                
                with torch.amp.autocast(enabled=self.config.fp16,device_type=self.device):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                if not loss.requires_grad:
                    # 如果不需要梯度，打印警告并跳过
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
                global_step += 1

                progress_bar.set_postfix({
                    'loss': f"{loss.item() * accumulation_steps:.4f}", 
                    'lr': f"{self.lr_scheduler.get_last_lr()[0]:.8f}"
                })


            avg_loss = epoch_loss / len(self.train_dataloader)
            print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
            if (epoch+1) % 5 == 0:
                self.save_model(epoch)
    
    def save_model(self, epoch):
        output_dir = os.path.join(self.config.output_dir, f"checkpoint-{epoch+1}")
        os.makedirs(output_dir, exist_ok=True)

        self.model.save_pretrained(output_dir)

        self.tokenizer.save_pretrained(output_dir)
        
        if self.config.save_full_model:
            merged_model = self.model.merge_and_unload()
            merged_model.save_pretrained(os.path.join(output_dir, "full_model"))
        
        print(f"Model saved to {output_dir}")