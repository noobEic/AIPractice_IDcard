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
        
        encoding = self.tokenizer(
            question,
            question + answer,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        labels = encoding['input_ids'].clone()
        answer_start_idx = len(self.tokenizer(question, return_tensors='pt')['input_ids'][0])
        labels[:, :answer_start_idx] = -100
        
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
        
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            torch_dtype=torch.bfloat16 if config.fp16 else torch.float32
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
        for epoch in range(self.config.num_train_epochs):
            print(f"Epoch {epoch+1}/{self.config.num_train_epochs}")
            
            epoch_loss = 0
            for step, batch in enumerate(tqdm(self.train_dataloader)):

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                with torch.amp.autocast(enabled=self.config.fp16,device_type=self.device):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss

                self.scaler.scale(loss).backward()

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                self.lr_scheduler.step()
                
                epoch_loss += loss.item()
                global_step += 1
                
                if global_step % 100 == 0:
                    print(f"Step {global_step}, Loss: {loss.item():.4f}, LR: {self.lr_scheduler.get_last_lr()[0]:.8f}")
            
            avg_loss = epoch_loss / len(self.train_dataloader)
            print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

            if (epoch + 1) % self.config.save_steps == 0:
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