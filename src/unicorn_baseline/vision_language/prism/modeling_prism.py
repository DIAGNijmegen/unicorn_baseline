import logging
from typing import Any, Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, einsum, nn
from transformers import PreTrainedModel
from transformers.generation.utils import GenerateOutput
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)

from unicorn_baseline.vision_language.prism.biogpt import BioGPT
from unicorn_baseline.vision_language.prism.configuring_prism import PrismConfig
from unicorn_baseline.vision_language.prism.perceiver import PerceiverResampler

logger = logging.getLogger(__file__)


class EmbedToLatents(nn.Module):
    def __init__(self, dim: int, dim_latents: int):
        super().__init__()

        self.to_latents = nn.Linear(dim, dim_latents, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        latents = self.to_latents(x)
        return F.normalize(latents, dim=-1)


PRISM_DOCSTRING = """
    Prism is a vision-language model for histopathology.

    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`~PrismConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
    """


PRISM_FORWARD_DOCSTRING = """
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.
            Call `self.tokenize` method to obtain token indices.

        tile_embeddings (`torch.FloatTensor` of shape `(batch_size, tile_sequence_length, tile_embedding_dim)`):
            Tile embeddings for a whole slide image from Virchow V1 model.

            See: https://huggingface.co/paige-ai/Virchow

        tile_mask (`torch.LongTensor` of shape `(batch_size, tile_sequence_length)`):
            Mask to avoid performing attention on padding tiles. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            To do batch processing on multiple whole slide images, pad each tile embedding sequence to the max sequence
            length in the batch and concatenate into a tensor.

            Necessary since whole slide images can have different number of tiles and therefore `tile_sequence_length`
            will be different for each image.

            ```python
            image_0 = torch.randn(10, 2560)  # WSI with 10 tiles
            image_1 = torch.randn(15, 2560)  # WSI with 15 tiles

            max_len = max([len(image_0), len(image_1)])

            tile_embeddings = torch.zeros((2, max_len, 2560))
            tile_embeddings[0, :len(image_0)] = image_0
            tile_embeddings[1, :len(image_1)] = image_1

            tile_mask = torch.zeros((2, max_len))
            tile_mask[0, :len(image_0)] = torch.ones(len(image_0))
            tile_mask[1, :len(image_1)] = torch.ones(len(image_1))
            ```

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


@add_start_docstrings(PRISM_DOCSTRING)
class Prism(PreTrainedModel):
    config_class = PrismConfig

    def __init__(self, config: PrismConfig):
        super().__init__(config)

        # Slide Encoder (Perceiver)

        self.image_resampler = PerceiverResampler(config.perceiver_config)

        # Text Decoder (BioGPT)

        self.text_decoder = BioGPT(
            config=config.biogpt_config,
            model_dir=config.model_dir,
            frozen_weights=config.biogpt_frozen_weights,
            frozen_embeddings=config.biogpt_frozen_embeddings,
            context_dim=config.biogpt_context_dim,
        )

        # Contrastive Head

        self.img_to_latents = EmbedToLatents(
            self.image_resampler.query_dim, config.dim_latents
        )
        self.text_to_latents = EmbedToLatents(
            self.text_decoder.text_dim, config.dim_latents
        )

        self.temperature = nn.Parameter(torch.log(torch.tensor(1.0) / 0.07))

        self.post_init()

    __call__: Callable[..., dict[str, Tensor]]

    @add_start_docstrings_to_model_forward(PRISM_FORWARD_DOCSTRING)
    def forward(
        self,
        input_ids: Tensor,
        tile_embeddings: Tensor,
        tile_mask: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        past_key_values: Optional[tuple[tuple[Tensor]]] = None,
        use_cache: bool = False,
    ) -> dict[str, Tensor]:
        # perceiver

        resampler_out = self.image_resampler(
            tile_embeddings=tile_embeddings,
            tile_mask=tile_mask,
        )

        # biogpt

        decoder_out = self.text_decoder(
            input_ids=input_ids,
            key_value_states=resampler_out["image_latents"],
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        # embeddings to projections on a hypersphere

        text_proj = self.text_to_latents(decoder_out["text_embedding"])
        image_proj = self.img_to_latents(resampler_out["image_embedding"])

        # text-image similarity

        sim = einsum("i d, j d -> i j", text_proj, image_proj)
        sim = sim * self.temperature.exp()
        assert sim.shape[0] == sim.shape[1]

        return {
            "logits": decoder_out["logits"],
            "text_embedding": decoder_out["text_embedding"],
            "image_embedding": resampler_out["image_embedding"],
            "image_latents": resampler_out["image_latents"],
            "sim": sim,
        }

    def tokenize(self, text: list[str]) -> Tensor:
        return self.text_decoder.tokenize(text)

    def untokenize(self, token_ids: Tensor) -> list[str]:
        return self.text_decoder.untokenize(token_ids)

    @torch.no_grad()  # type: ignore
    def generate(
        self,
        inputs: Optional[Tensor] = None,
        key_value_states: Optional[Tensor] = None,
        max_length: int = 100,
        do_sample: bool = False,
        num_beams: int = 1,
        num_beam_groups: int = 1,
        **kwargs: Any,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        """
        Args:
            inputs: input token indices used as prompt; leave as default (`None`) or
                use `"Diagnosis: "` string as prompt.
                Call `self.tokenize` to tokenize a batch of strings into indices.
                shape: (batch size, sequence length)

            key_value_states: image latents from Perceiver.
                Call `self.image_resampler(tile_embeddings, tile_mask)['image_latents']` to generate image latents.
                shape: (batch size, 512, 1280)

            max_length: the maximum length the generated tokens can have.

            do_sample: whether or not to use sampling; use greedy decoding otherwise.

            num_beams: number of beams for beam search. 1 means no beam search.

            num_beam_groups: number of groups to divide `num_beams` into in order to ensure
                diversity among different groups of beams.
                [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details.
        """
        if key_value_states is None:
            raise Exception(
                "image latents (key_value_states) are required for generation."
            )

        batch_size = len(key_value_states)
        device = key_value_states.device

        if inputs is None:
            inputs = torch.tensor(
                [[self.text_decoder.bos_token_id]] * batch_size, device=device
            )

        return self.text_decoder.model.generate(
            inputs=inputs,
            key_value_states=key_value_states,
            max_length=max_length,
            do_sample=do_sample,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            **kwargs,
        )

    @torch.no_grad()  # type: ignore
    def zero_shot(
        self,
        image_embeds: Tensor,
        *,
        neg_prompts: list[str],
        pos_prompts: list[str],
    ) -> Tensor:
        """
        Args:
            image_embeds: slide embedding per image.
                Call `self.image_resampler(tile_embeddings, tile_mask)['image_embedding']` to generate slide embeddings.
                shape: (image batch size, 1280)

            neg_prompts: list of prompts that are attributed to the negative class.
                If more than one is provided, their probability scores will be marginalised over.

            pos_prompts: list of prompts that are attributed to the positive class.
                If more than one is provided, their probability scores will be marginalised over.

        Returns:
            Probability values for the negative class and the positive class, per each image.
            shape: (image batch size, 2)
        """
        device = image_embeds.device

        # zero-shot prompts

        # (Bn + Bp, N)
        zero_shot_prompts = neg_prompts + pos_prompts
        zero_shot_token_ids = self.tokenize(zero_shot_prompts)[:, :-1]
        zero_shot_token_ids = zero_shot_token_ids.to(device)

        # (Bn + Bp, M)
        dummy_image_latents = torch.empty(
            (len(zero_shot_prompts), 1, self.text_decoder.context_dim), device=device
        )

        decoder_out = self.text_decoder(zero_shot_token_ids, dummy_image_latents)

        # zero-shot probabilities

        text_proj = self.text_to_latents(decoder_out["text_embedding"])
        image_proj = self.img_to_latents(image_embeds)

        # (Bi, Bn + Bp)
        sim = einsum("i d, j d -> i j", image_proj, text_proj)  # (image, text)
        sim = sim * self.temperature.exp()

        assert sim.shape[0] == len(image_embeds)
        assert sim.shape[1] == len(zero_shot_prompts)

        zero_shot_probs = torch.softmax(sim.to(torch.float), dim=-1)

        # (Bi, 2)
        neg_pos_probs = torch.cat(
            [
                zero_shot_probs[:, : len(neg_prompts)].sum(-1, keepdim=True),
                zero_shot_probs[:, len(neg_prompts) :].sum(-1, keepdim=True),
            ],
            dim=-1,
        )

        return neg_pos_probs

    def slide_representations(
        self,
        tile_embeddings: Tensor,
        tile_mask: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        """
        Args:
            tile_embeddings: sequences of tile embeddings per whole slide image
                shape: (batch size, sequence length, feature dim)
            attn_mask: mask out padded area for each sequence in the batch;
                necessary for batch size > 1 since images can have different sequence lengths.
                shape: (batch size, sequence length)

        Returns:
            dict with image embedding and image latents per sequence
        """
        return self.image_resampler(tile_embeddings, tile_mask=tile_mask)
