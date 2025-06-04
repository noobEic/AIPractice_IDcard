from train.trainer import IDCardTrainer, IDCardConfig, IDCardDataset
from common.utils import set_seed
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

if __name__ == '__main__':
    train_config = IDCardConfig(
        model_name_or_path='/home/zxyu/private_data/pretrain/Qwen2.5-3B-Instruct',
        tokenizer_name_or_path='/home/zxyu/private_data/pretrain/Qwen2.5-3B-Instruct',
        data_path='./synthetic_data_generation_ragas_2.csv',
        output_dir='./output',
        max_length=512, 
        batch_size=1,
        learning_rate=2e-5,
        num_train_epochs=50,
        seed=42,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        fp16=True,
    )
    
    dataset = IDCardDataset(train_config.data_path, train_config.tokenizer_name_or_path, train_config.max_length)
    trainer = IDCardTrainer(train_config)
    set_seed(train_config.seed)
    trainer.train()
    