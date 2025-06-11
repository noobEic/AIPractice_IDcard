from train.trainer import IDCardTrainer, IDCardConfig, IDCardDataset
from common.utils import set_seed
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == '__main__':
    train_config = IDCardConfig(
        model_name_or_path='/home/zxyu/private_data/pretrain/Qwen2.5-7B-Instruct',
        tokenizer_name_or_path='/home/zxyu/private_data/pretrain/Qwen2.5-7B-Instruct',
        data_path='./synthetic_data_generation_ragas_2.csv',
        output_dir='./output/Qwen2.5-7B-Instruct',
        max_length=512, 
        batch_size=1,
        learning_rate=2e-5,
        num_train_epochs=5,
        seed=42,
        device='cuda:1' if torch.cuda.is_available() else 'cpu',
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        fp16=True,
        accumulation_steps=4,
        save_steps=100,
        save_full_model=False,
    )
    
    dataset = IDCardDataset(train_config.data_path, train_config.tokenizer_name_or_path, train_config.max_length)
    trainer = IDCardTrainer(train_config)
    set_seed(train_config.seed)
    trainer.train()
    