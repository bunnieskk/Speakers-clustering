import torch
from transformers import WavLMModel


def masked_pool(hidden: torch.Tensor, attention_mask: torch.Tensor, pooling: str) -> torch.Tensor:
    """
    hidden: [B, T, H]
    attention_mask: [B, T] (1 valid, 0 pad)
    pooling: "mean" or "stats"
    return:
      - mean:  [B, H]
      - stats: [B, 2H]  (mean + std)
    """
    m = attention_mask.unsqueeze(-1).to(hidden.dtype)         
    denom = m.sum(dim=1).clamp(min=1.0)                       
    mean = (hidden * m).sum(dim=1) / denom               

    if pooling == "mean":
        return mean

    var = ((hidden - mean.unsqueeze(1)) ** 2 * m).sum(dim=1) / denom
    std = torch.sqrt(var.clamp(min=1e-8))
    return torch.cat([mean, std], dim=-1)


class WavLMEmbedder:
    """
    Minimal WavLM embedder for speaker clustering.
    Input: batch from your dataloader
      - input_values: [B, T]
      - attention_mask: [B, T]
    Output:
      - embeddings for chosen layers
    """

    def __init__(self, model_name: str, device: str, cache_dir: str | None = None):
        self.device = device
        self.model = WavLMModel.from_pretrained(model_name, cache_dir=cache_dir).to(device)
        self.model.eval()

    @torch.no_grad()
    def forward_layers(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor,
        layers: list[int],
        pooling: str = "mean",
        l2_normalize: bool = True,
    ) -> dict[int, torch.Tensor]:
        """
        layers: e.g. list(range(8, 21))
        returns: {layer: embedding_tensor}
          - embedding_tensor: [B, H] or [B, 2H] on CPU
        """
        input_values = input_values.to(self.device)
        attention_mask = attention_mask.to(self.device)

        out = self.model(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        layer_to_emb = {}
        for layer in layers:
            hidden = out.hidden_states[layer]                           # [B,T,H]
            emb = masked_pool(hidden, attention_mask, pooling=pooling)  # [B,H] or [B,2H]
            if l2_normalize:
                emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
            layer_to_emb[layer] = emb.detach().cpu()

        return layer_to_emb

    @torch.no_grad()
    def forward_one(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor,
        layer: int,
        pooling: str = "mean",
        l2_normalize: bool = True,
    ) -> torch.Tensor:
        """Convenience wrapper for a single layer."""
        return self.forward_layers(
            input_values=input_values,
            attention_mask=attention_mask,
            layers=[layer],
            pooling=pooling,
            l2_normalize=l2_normalize,
        )[layer]
