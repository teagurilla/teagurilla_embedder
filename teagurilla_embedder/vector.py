from typing import AsyncIterable, AsyncIterator, Iterable
import torch
from transformers import AutoModel
from transformers.tokenization_utils_base import BatchEncoding
from transformers.modeling_outputs import BaseModelOutput


class Vectorizer:
    """
    Simple vectorizer class.

    Arrtibutes:
        model: AutoModel - the transformers library auto tokenizer
    """

    def __init__(self, checkpoint: str = "roberta-base"):
        """
        Init for Tokenizer.

        Args:
            checkpoint: the model checkpoint name.
        """
        self.model = AutoModel.from_pretrained(
            checkpoint, output_hidden_states=True
        )

    def vectorize(self, tokens: BatchEncoding) -> torch.Tensor:
        """
        Vectorize the tokens

        Args:
            tokens: BatchEncoding - tokenized sentence (or multiple sentences)

        Returns:
            BaseModelOutput - the... stuff the model returns?
        """
        return self.ndim_tenson_into_vector(self.model(**tokens).hidden_states)

    @staticmethod
    def ndim_tenson_into_vector(
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apparently, the model gives a tensor of that's something like
        [22 x 12 x 768], and we need a [768], so this one just averages it all

        Args:
            tensor: tuple[torch.Tensor] - a singleton of torch.Tensor
        """
        token_vecs = tensor[-2][0]
        return torch.mean(token_vecs, dim=0)
