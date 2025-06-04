from train.trainer import IDCardTrainer, IDCardConfig, IDCardDataset
from common.utils import set_seed
import torch
if __name__ == '__main__':
    train_config = IDCardConfig(
        model_name_or_path='/home/zxyu/private_data/pretrain/Qwen2.5-3B-Instruct',
        tokenizer_name_or_path='/home/zxyu/private_data/pretrain/Qwen2.5-3B-Instruct',
        data_path='./synthetic_data_generation_ragas_2.csv',
        output_dir='./output',
        max_length=512, 
        batch_size=16,
        learning_rate=2e-5,
        num_train_epochs=3,
        seed=42,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    dataset = IDCardDataset(train_config.data_path, train_config.tokenizer_name_or_path, train_config.max_length)
    trainer = IDCardTrainer(train_config)
    set_seed(train_config.seed)
    trainer.train(dataset)
    