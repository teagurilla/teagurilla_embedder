"""
The embdeer module.
The embedder generates vectors for sentences.
Those embeddings are then used for searching the vector DB,
which is not yet ready
"""

import torch
from typing import AsyncIterable, AsyncIterator, Iterable, Iterator
from .tokenizer import Tokenizer
from .vector import Vectorizer


class Embedder:
    """
    The embedder class.


    """

    def __init__(self, checkpoint: str = "roberta-base"):
        """
        Init for the embedder class

        Args:
            checkpoint - the model checkpoint name.
        """

        self.tokenizer = Tokenizer(checkpoint)
        self.vectorizer = Vectorizer(checkpoint)

    def sentence_into_vector(self, sentence: str) -> torch.Tensor:
        """
        Vectorize a single sentence

        Args:
            sentence: str - the sentence

        Returns:
            torch.Tensor - the vector representing the sentence
        """

        return self.vectorizer.vectorize(
            self.tokenizer.tokens_from_str(sentence)
        )

    def sentence_list_into_vectors(
        self, sentences: Iterable[str]
    ) -> Iterator[torch.Tensor]:
        """
        Vectorize a single sentence

        Args:
            sentence: Iterable[str] - the sentence iterable

        Returns:
            Iterator[torch.Tensor] - the vector iterator representing the sentences
        """
        for sentence in sentences:
            yield self.vectorizer.vectorize(
                self.tokenizer.tokens_from_str(sentence)
            )

    async def sentence_list_into_vectors_async(
        self, sentences: AsyncIterable[str]
    ) -> AsyncIterator[torch.Tensor]:
        """
        Vectorize a single sentence

        Args:
            sentence: AsyncIterable[str] - the sentence iterable

        Returns:
            AsyncIterator[torch.Tensor] - the vector iterator representing the sentences
        """
        async for sentence in sentences:
            yield self.vectorizer.vectorize(
                self.tokenizer.tokens_from_str(sentence)
            )
