import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
from tqdm import tqdm
import csv
import os
from dataclasses import dataclass
from common.utils import set_seed

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
        text = f"问题：{item['question']}"
        
        encoding = self.tokenizer(
            text,
            item['answer'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(item['label'], dtype=torch.long)
        }
    
class IDCardTrainer:
    def __init__(self,config:IDCardConfig):
        self.model_name_or_path = config.model_name_or_path
        self.tokenizer_name_or_path = config.tokenizer_name_or_path
        self.data_path = config.data_path
        self.output_dir = config.output_dir
        self.max_length = config.max_length
        self.batch_size = config.batch_size
        self.learning_rate = config.learning_rate
        

        self.num_train_epochs = config.num_train_epochs
        self.seed = config.seed
        self.device = config.device
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path).to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        self.train_dataset = IDCardDataset(self.data_path, self.tokenizer, self.max_length)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        set_seed(self.seed)
    def train(self):
        os.makedirs(self.output_dir, exist_ok=True)
        self.model.train()
        for epoch in range(self.num_train_epochs):
            print(f"Epoch {epoch+1} / {self.num_train_epochs}")
            for step, batch in enumerate(tqdm(self.train_dataloader)):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            self.save_model(epoch)

    def save_model(self, epoch):
        output_dir = os.path.join(self.output_dir, f"epoch_{epoch+1}")
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)