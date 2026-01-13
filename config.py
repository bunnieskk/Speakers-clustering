"""
配置文件 - 使用 WavLM-Large 提取 Speaker Embedding）
"""
import os
import torch
from dataclasses import dataclass

@dataclass
class Config:
    """WavLM Speaker Embedding 配置"""

    # ========= 模型 =========
    model_name: str = "microsoft/wavlm-large"
    cache_dir: str = "./cache"

    # ========= 音频 =========
    # WavLM 统一使用 16kHz
    sample_rate: int = 16000

    # ========= Embedding 设置 =========
    # WavLM Large: 24 层 Transformer
    # speaker 信息主要集中在中间层
    embedding_layers: list[int] = list(range(8, 21))  # [8, 9, ..., 20]选取中间层
    pooling: str = "mean"          # "mean" 或 "stats"
    l2_normalize: bool = True     
    # ========= 推理 =========
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    embed_batch_size: int = 8      # WavLM Large 吃显存

    # ========= 输出 / 复现 =========
    output_dir: str = "./outputs"
    seed: int = 42

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

        print("配置参数 (WavLM Speaker Embedding):")
        for k, v in self.__dict__.items():
            print(f"  {k}: {v}")


config = Config()
