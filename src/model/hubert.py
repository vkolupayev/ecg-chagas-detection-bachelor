import logging
import json
from pathlib import Path

import torch.nn as nn
from collections import OrderedDict
from transformers import AutoModel

from src.model.mrfcnn import LargeBlock, SmallBlock, init_weights
from src.model.hubert_ecg import HuBERTECGConfig, HuBERTECG

logger = logging.getLogger(__name__)


class ChagasHuBERT(nn.Module):
    """
    HuBERT-ECG.
    Authors: Edoardo Coppola1, Mattia Savardi, Mauro Massussi, Marianna Adamo, Marco Metra, Alberto Signoroni.
    Title: HuBERT-ECG as a self-supervised foundation model for broad and scalable cardiac applications.
    https://www.medrxiv.org/content/10.1101/2024.11.14.24317328v3
    """

    def __init__(
        self,
        simple_classifier: bool = True,
        pretrained: bool = True,
        model_path: str | Path | None = None,
        model_size: str | None = None,
    ):
        assert model_size in [
            "small",
            "base",
            "large",
            None,
        ], 'Invalid model size. Possible options: ["small", "base", "large"] or None if loading a local model.'
        assert (
            model_path or model_size
        ), "A local model path or the model size of the remote repository must be specified."
        super().__init__()

        if pretrained:
            if model_path:
                self.hubert_ecg = AutoModel.from_pretrained(
                    model_path, trust_remote_code=True
                )
            else:
                model_repo_name = f"Edoardo-BS/hubert-ecg-{model_size}"
                self.hubert_ecg = AutoModel.from_pretrained(
                    model_repo_name, trust_remote_code=True
                )
        else:
            with open(f"{model_path}/config.json", "r") as f:
                hubert_ecg_config = json.load(f)
            hubert_ecg_config = HuBERTECGConfig(**hubert_ecg_config)

            self.hubert_ecg = HuBERTECG(hubert_ecg_config)

        del self.hubert_ecg.label_embedding  # not needed
        del self.hubert_ecg.final_proj  # not needed
        # as we load pre-trained models that used to mask inputs, resetting masking probs prevents masking
        self.hubert_ecg.config.mask_time_prob = 0.0
        self.hubert_ecg.config.mask_feature_prob = 0.0
        if simple_classifier:
            self.classifier = SimpleHead(self.hubert_ecg.config.hidden_size, 1)
        else:
            self.classifier = HeadMRFCNN(self.hubert_ecg.config.hidden_size, 1)

    def forward(self, *args, **kwargs):
        outputs = self.hubert_ecg(*args, **kwargs).last_hidden_state
        logits = self.classifier(outputs)
        return logits


class HeadMRFCNN(nn.Module):
    def __init__(self, input_channels=512, num_classes=1):
        super(HeadMRFCNN, self).__init__()

        self.large_block = LargeBlock(input_channels, input_channels)
        self.conv1 = nn.Conv1d(
            input_channels * 5,
            input_channels * 5,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.small_block1 = SmallBlock(input_channels * 5, input_channels * 2)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(input_channels * 2 * 3, num_classes)
        self.apply(init_weights)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.large_block(x)
        x = self.conv1(x)
        x = self.small_block1(x)
        x = self.global_avg_pool(x).squeeze(-1)
        x = self.fc(x)
        return x


class SimpleHead(nn.Module):
    def __init__(self, input_channels=512, num_classes=1):
        super(SimpleHead, self).__init__()
        self.classifier = nn.Sequential(
            OrderedDict(
                [
                    ("drop_out", nn.Dropout(0.1)),
                    ("linear", nn.Linear(input_channels, num_classes)),
                ]
            )
        )
        self.apply(init_weights)

    def forward(self, x):
        x = x.mean(dim=1)
        return self.classifier(x)


import torch
import torch.nn as nn
from transformers import HubertConfig, HubertModel
from typing import List


class HuBERTECGConfig(HubertConfig):

    model_type = "hubert_ecg"

    def __init__(
        self, ensemble_length: int = 1, vocab_sizes: List[int] = [100], **kwargs
    ):
        super().__init__(**kwargs)
        self.ensemble_length = ensemble_length
        self.vocab_sizes = (
            vocab_sizes if isinstance(vocab_sizes, list) else [vocab_sizes]
        )


class HuBERTECG(HubertModel):

    config_class = HuBERTECGConfig

    def __init__(self, config: HuBERTECGConfig):
        super().__init__(config)
        self.config = config

        self.pretraining_vocab_sizes = config.vocab_sizes

        assert config.ensemble_length > 0 and config.ensemble_length == len(
            config.vocab_sizes
        ), f"ensemble_length {config.ensemble_length} must be equal to len(vocab_sizes) {len(config.vocab_sizes)}"

        # final projection layer to map encodings into the space of the codebook
        self.final_proj = nn.ModuleList(
            [
                nn.Linear(config.hidden_size, config.classifier_proj_size)
                for _ in range(config.ensemble_length)
            ]
        )

        # embedding for codebooks
        self.label_embedding = nn.ModuleList(
            [
                nn.Embedding(vocab_size, config.classifier_proj_size)
                for vocab_size in config.vocab_sizes
            ]
        )

        assert len(self.final_proj) == len(
            self.label_embedding
        ), f"final_proj and label_embedding must have the same length"

    def logits(self, transformer_output: torch.Tensor) -> torch.Tensor:
        # takes (B, T, D)

        # compute a projected output for each ensemble
        projected_outputs = [
            final_projection(transformer_output) for final_projection in self.final_proj
        ]

        ensemble_logits = [
            torch.cosine_similarity(
                projected_output.unsqueeze(2),
                label_emb.weight.unsqueeze(0).unsqueeze(0),
                dim=-1,
            )
            / 0.1
            for projected_output, label_emb in zip(
                projected_outputs, self.label_embedding
            )
        ]

        return ensemble_logits  # returns [(BS, T, V)] * ensemble_length
