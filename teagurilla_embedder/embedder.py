"""The embdeer module.
The embedder generates vectors for sentences.
Those embeddings are then used for searching the vector DB,
which is not yet ready
"""

from typing import AsyncIterable, AsyncIterator, Iterable, Iterator

from sentence_transformers import SentenceTransformer, util
import torch


class Embedder:
    """The embedder class."""

    def __init__(self, checkpoint: str = "sentence-transformers/all-roberta-large-v1"):
        """Init for the embedder class

        Args:
            checkpoint - the model checkpoint name.
        """

        self.model = SentenceTransformer(checkpoint)

    def sentence_into_vector(self, sentence: str) -> torch.Tensor:
        """Vectorize a single sentence

        Args:
            sentence: str - the sentence

        Returns:
            torch.Tensor - the vector representing the sentence
        """

        return self.model.encode(sentence)

    def sentence_iter_into_vectors(
        self, sentences: Iterable[str]
    ) -> Iterator[torch.Tensor]:
        """Vectorize an iterable

        Args:
            sentence: Iterable[str] - the sentence iterable

        Returns:
            Iterator[torch.Tensor] - the vector iterator representing the sentences
        """
        for sentence in sentences:
            yield self.model.encode(sentence)

    def sentence_list_into_vectors(self, sentences: list[str]) -> list[torch.Tensor]:
        """Vectorize a list of sentences
        NOTE: does not work

        Args:
            sentences: list[str] - the sentence list

        Returns:
            list[torch.Tensor] - the list of embeddings
        """
        # TODO: make this work
        return self.model.encode(sentences)

    async def sentence_list_into_vectors_async(
        self, sentences: AsyncIterable[str]
    ) -> AsyncIterator[torch.Tensor]:
        """Vectorize a single sentence

        Args:
            sentence: AsyncIterable[str] - the sentence iterable

        Returns:
            AsyncIterator[torch.Tensor] - the vector iterator representing the sentences
        """
        async for sentence in sentences:
            yield self.model.encode(sentence)
