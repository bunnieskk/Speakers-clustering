#Speaker Clustering DataLoader (Embedding stage)

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
import torchaudio

from datasets import load_dataset  


@dataclass
class DataConfig:
    # Audio processing
    sample_rate: int = 16000
    max_duration_sec: float = 10.0  # 先截断，后续再做 VAD/segmentation

    # HF datasets loading
    dataset_name: str = "MLCommons/peoples_speech"
    dataset_config: Optional[str] = None
    split: str = "train"
    streaming: bool = True
    cache_dir: str = "./cache"

    # DataLoader
    batch_size: int = 8
    num_workers: int = 0
    pin_memory: bool = True

    # Field mapping
    audio_field: str = "audio"
    id_field: Optional[str] = None  # None: 用 sample.get("id") 或 idx


def _ensure_mono(wav: torch.Tensor) -> torch.Tensor:
    # wav: [T] or [C, T] -> [T]
    if wav.dim() == 1:
        return wav
    if wav.dim() == 2:
        return wav.mean(dim=0)
    raise ValueError(f"Unexpected waveform shape: {tuple(wav.shape)}")


def _resample_if_needed(wav: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    if orig_sr == target_sr:
        return wav
    return torchaudio.functional.resample(wav, orig_sr, target_sr)


def _truncate(wav: torch.Tensor, sr: int, max_sec: float) -> torch.Tensor:
    if max_sec is None or max_sec <= 0:
        return wav
    max_len = int(sr * max_sec)
    return wav[:max_len] if wav.numel() > max_len else wav


class SpeakerAudioDataset(TorchDataset):
    """
    Dataset for WavLM embedding extraction (speaker clustering pre-step).

    Returns item:
      audio: FloatTensor [T] (mono, resampled, truncated)
      id:    any
    """

    def __init__(self, cfg: DataConfig, max_items: Optional[int] = None):
        self.cfg = cfg
        self.max_items = max_items

        # 注意：People's Speech 体量巨大，强烈建议 streaming=True
        self.ds = load_dataset(
            cfg.dataset_name,
            cfg.dataset_config,
            split=cfg.split,
            streaming=cfg.streaming,
            cache_dir=cfg.cache_dir,
        )

        if cfg.streaming:
            # streaming 没有 __len__，用 max_items 控制可迭代数量
            if max_items is None:
                raise ValueError("streaming=True 时必须传 max_items（例如 1000）以获得可用的 __len__/__getitem__。")

            self._materialized: List[Dict[str, Any]] = []
            for i, ex in enumerate(self.ds):
                self._materialized.append(ex)
                if i + 1 >= max_items:
                    break
            self.ds = self._materialized

        # 非 streaming 或已 materialize
        self._len = len(self.ds)
        if (not cfg.streaming) and max_items is not None:
            self._len = min(self._len, max_items)

    def __len__(self):
        return self._len

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.ds[idx]

        audio = ex[self.cfg.audio_field]
        # HF audio feature: {"array": np.ndarray, "sampling_rate": int}
        wav = torch.tensor(audio["array"], dtype=torch.float32)
        sr = int(audio.get("sampling_rate", self.cfg.sample_rate))

        wav = _ensure_mono(wav)
        wav = _resample_if_needed(wav, sr, self.cfg.sample_rate)
        wav = _truncate(wav, self.cfg.sample_rate, self.cfg.max_duration_sec)

        if self.cfg.id_field and self.cfg.id_field in ex:
            ex_id = ex[self.cfg.id_field]
        else:
            ex_id = ex.get("id", idx)

        return {"audio": wav, "id": ex_id}


def collate_audio(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    audios = [b["audio"] for b in batch]
    ids = [b["id"] for b in batch]
    lengths = torch.tensor([a.numel() for a in audios], dtype=torch.long)

    padded = torch.nn.utils.rnn.pad_sequence(audios, batch_first=True, padding_value=0.0)
    B, T = padded.shape
    arange = torch.arange(T).unsqueeze(0).expand(B, T)
    attention_mask = (arange < lengths.unsqueeze(1)).long()

    return {
        "input_values": padded,          # [B, T]
        "attention_mask": attention_mask, # [B, T]
        "lengths": lengths,              # [B]
        "ids": ids,                      # list
    }


def build_dataloader(cfg: DataConfig, max_items: int) -> DataLoader:
    ds = SpeakerAudioDataset(cfg, max_items=max_items)
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=not cfg.streaming,  # streaming/materialize 也可以 shuffle
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=collate_audio,
    )
