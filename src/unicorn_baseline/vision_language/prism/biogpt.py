from typing import Any, Callable, Optional, Type, TypeVar

import torch
from torch import Tensor, nn
from pathlib import Path
from transformers import BioGptTokenizer
from transformers.models.biogpt.configuration_biogpt import BioGptConfig

from unicorn_baseline.vision_language.prism.biogpt_hf import BioGptForCausalLM


class BioGPT(nn.Module):
    def __init__(
        self,
        config: BioGptConfig,
        model_dir: Path,
        frozen_weights: bool = True,
        frozen_embeddings: bool = False,
        context_dim: int = 1280,
    ):
        super().__init__()

        self.context_dim = context_dim

        self.model = BioGptForCausalLM(config, x_attn_dim=context_dim)

        commit_hash = "eb0d815e95434dc9e3b78f464e52b899bee7d923"
        tokenizer_weights = model_dir / "biogpt"
        self.tokenizer = enforce_type(
            BioGptTokenizer,
            BioGptTokenizer.from_pretrained(tokenizer_weights, revision=commit_hash),
        )

        self.text_dim = enforce_type(
            torch.nn.Embedding, self.model.biogpt.embed_tokens
        ).embedding_dim

        # choose cls token from unused tokens
        # token 42383 : "madeupword0006</w>"
        # see https://huggingface.co/microsoft/biogpt/raw/main/vocab.json
        self.cls_token_id = 42383
        # init to random weight from the same distribution as original embeddings init
        enforce_type(Tensor, self.model.biogpt.embed_tokens.weight)[
            self.cls_token_id
        ].data.normal_(mean=0.0, std=self.model.config.initializer_range)

        if frozen_weights:
            for name, param in self.model.named_parameters():
                if not any(c in name for c in ["embed_tokens", "x_attn"]):
                    param.requires_grad = False

        if frozen_embeddings:
            for name, param in self.model.named_parameters():
                if "embed_tokens" in name:
                    param.requires_grad = False

    @property
    def pad_id(self) -> int:
        return enforce_type(int, self.tokenizer.pad_token_id)

    @property
    def bos_token_id(self) -> int:
        # biogpt uses EOS as BOS
        return enforce_type(int, self.tokenizer.eos_token_id)

    @property
    def eos_token_id(self) -> int:
        return enforce_type(int, self.tokenizer.eos_token_id)

    def tokenize(self, text: list[str]) -> Tensor:
        # add EOS token
        text = [t + self.tokenizer.eos_token for t in text]

        text_tokenised = self.tokenizer(
            text=text,
            add_special_tokens=True,
            padding=True,
            return_tensors="pt",
            return_token_type_ids=False,
            return_attention_mask=False,
        )

        # example:
        # "</s>Squamous mucosa indicative of reflux esophagitis. </s>"
        # [    2, 21477,  2626,  7265,     5,  4532, 13893,     4,     2]
        text_token_ids = enforce_type(Tensor, text_tokenised["input_ids"])
        return text_token_ids

    def untokenize(self, token_ids: Tensor) -> list[str]:
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=False)

    __call__: Callable[..., dict[str, Tensor]]

    def forward(
        self,
        input_ids: Tensor,
        key_value_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        past_key_values: Optional[tuple[tuple[Tensor]]] = None,
        use_cache: bool = False,
    ) -> dict[str, Tensor]:
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
                Call `self.tokenize` method to obtain token indices.

            key_value_states (`torch.FloatTensor` of shape `(batch_size, context_length, context_dim)`):
                Context for cross-attention modules in BioGPT; specifically, image latents from Perceiver.

            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                If not provided, the mask will be computed using positions of pad token in `input_ids`.

            head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
                Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
                `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
                `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
                blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
                don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
                `decoder_input_ids` of shape `(batch_size, sequence_length)`. inputs_embeds (`torch.FloatTensor` of shape
                `(batch_size, sequence_length, hidden_size)`, *optional*): Optionally, instead of passing `input_ids` you
                can choose to directly pass an embedded representation. This is useful if you want more control over how to
                convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
                `past_key_values`).
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # append cls token id
        cls_token_ids = torch.full((batch_size, 1), self.cls_token_id, device=device)
        input_ids = torch.cat([input_ids, cls_token_ids], dim=-1)

        # attention mask
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_id).to(torch.int8)

        output = self.model(
            input_ids=input_ids,
            key_value_states=key_value_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=True,
            xattn_collect_attn=None,
            return_dict=True,
        )

        # remove multimodal cls token
        logits = output["logits"][:, :-1]

        # take unimodal cls token from the last unimodal layer
        last_unimodal_layer = self.model.biogpt.first_x_attn_layer - 1
        # NOTE: `+ 1` accounts for the 0th entry in hidden_states being
        # not a layer output but text embeddings
        text_cls_embedding = output["hidden_states"][last_unimodal_layer + 1][:, -1]

        return {"logits": logits, "text_embedding": text_cls_embedding}


T = TypeVar("T")


def enforce_type(t: Type[T], o: Any) -> T:
    if not isinstance(o, t):
        raise TypeError(f"{type(o)=} != t")
    return o
