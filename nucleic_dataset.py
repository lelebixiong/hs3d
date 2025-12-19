import os
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel


class NucleicAcidDataset(Dataset):
    def __init__(self, 
                 data_dir, 
                 max_length=256,
                 use_teacher=True,
                 teacher_model_name="InstaDeepAI/nucleotide-transformer-500m-human-ref",
                 device=None):
        
        self.device = device
        
        file_path = os.path.join(data_dir, "train1.npz")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件不存在: {file_path}")

        data = np.load(file_path, allow_pickle=True)
        self.sequences = data["sequences"]  
        self.max_length = max_length
        self.use_teacher = use_teacher

        print(f"加载数据: {len(self.sequences)} 条序列, 最大长度 {self.max_length}")
        print(f"使用设备: {self.device}")
        
        # 始终加载教师模型的 tokenizer
        print(f"加载 tokenizer: {teacher_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            teacher_model_name,
            trust_remote_code=True
        )
        print(f"Tokenizer 词汇表大小: {self.tokenizer.vocab_size}")
        
        if use_teacher:
            print(f"加载教师模型: {teacher_model_name}")
            self.teacher_model = AutoModel.from_pretrained(
                teacher_model_name,
                use_safetensors=True,    
            ).to(self.device)
            self.teacher_model.eval() 
            
            self.teacher_hidden_size = self.teacher_model.config.hidden_size
            print(f"教师模型特征维度: {self.teacher_hidden_size}")
        else:
            self.teacher_model = None
            self.teacher_hidden_size = None
        
        print("✓ 使用教师模型的 tokenizer 进行编码")

    def __len__(self):
        return len(self.sequences)

    def encode_sequence(self, sequence):
        # 清理序列：只保留有效字符
        valid_chars = set('ATUCGNatucgn')
        cleaned_seq = ''.join([c for c in sequence if c in valid_chars])
        
        if len(cleaned_seq) < 1:
            # 空序列：返回全PAD
            input_ids = torch.full((self.max_length,), self.tokenizer.pad_token_id, dtype=torch.long)
            attention_mask = torch.zeros(self.max_length, dtype=torch.float)
            return input_ids, attention_mask
        
        # 使用教师 tokenizer 编码
        encoded = self.tokenizer(
            cleaned_seq,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoded["input_ids"].squeeze(0)  # [max_length]
        attention_mask = encoded["attention_mask"].squeeze(0).float()  # [max_length]
        
        return input_ids, attention_mask
    
    def extract_teacher_features(self, sequence):
        """提取教师模型的特征"""
        if not self.use_teacher:
            return None
        
        # 清理序列
        valid_chars = set('ATUCGNatucgn')
        cleaned_seq = ''.join([c for c in sequence if c in valid_chars])
        
        if len(cleaned_seq) < 1:
            return torch.zeros(self.teacher_hidden_size)
        
        # 使用相同的 tokenizer 编码
        encoded = self.tokenizer(
            cleaned_seq,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        with torch.no_grad():
            outputs = self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            # 使用 [CLS] token 的表示
            teacher_embeddings = outputs.last_hidden_state[:, 0, :].squeeze(0)
        
        return teacher_embeddings.cpu()

    def __getitem__(self, idx):
        sequence = str(self.sequences[idx])
        input_ids, attention_mask = self.encode_sequence(sequence)
        
        if self.use_teacher:
            teacher_embeddings = self.extract_teacher_features(sequence)
            return input_ids, attention_mask, teacher_embeddings
        else:
            return input_ids, attention_mask