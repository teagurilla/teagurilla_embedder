from typing import AsyncIterable, AsyncIterator, Iterable
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding


class Tokenizer:
    """
    Simple Tokenizer class.

    Arrtibutes:
        tokenizer: AutoTokenizer - the transformers library auto tokenizer
    """

    def __init__(self, checkpoint: str = "roberta-base"):
        """
        Init for Tokenizer.

        Args:
            checkpoint: the model checkpoint name.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def tokens_from_str(
        self,
        sentence: str,
        padding: bool = True,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> BatchEncoding:
        """
        Tokenize a string

        Args:
            sentence: str - the sentence itself
            padding: bool - whether to pad input to 512 chars
            truncation: bool - whether to truncate input down to 512 chars
            return_tensors: str - type of return tensors

        Returns:
            BatchEncoding - the tokenized string
        """
        return self.tokenizer(
            sentence,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        )

    def tokens_from_iter(
        self,
        sentences: Iterable[str],
        padding: bool = True,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> BatchEncoding:
        """
        Tokenize a string

        Args:
            sentences: Iterable[str] - the sentences iterable
            padding: bool - whether to pad input to 512 chars
            truncation: bool - whether to truncate input down to 512 chars
            return_tensors: str - type of return tensors

        Returns:
            BatchEncoding - the tokenized strings
        """
        return self.tokenizer(
            sentences,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        )

    async def tokens_iter_from_iter(
        self,
        sentences: AsyncIterable[str],
        padding: bool = True,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> AsyncIterator[BatchEncoding]:
        """
        Tokenize a collection of strings, and return an iterator.
        I mean, telegram stuff does async fetching, so maybe use it there?

        Args:
            sentences: Iterable[str] - the sentences iterable
            padding: bool - whether to pad input to 512 chars
            truncation: bool - whether to truncate input down to 512 chars
            return_tensors: str - type of return tensors

        Returns:
            AsyncIterator[BatchEncoding] - the tokenized strings
        """
        async for sentence in sentences:
            yield self.tokenizer(
                sentence,
                padding=padding,
                truncation=truncation,
                return_tensors=return_tensors,
            )
