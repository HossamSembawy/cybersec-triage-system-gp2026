"""
Email Module — Preprocessor
Tokenisation and input preparation for DistilBERT inference.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from transformers import DistilBertTokenizerFast

logger = logging.getLogger(__name__)

TOKENIZER_NAME = "distilbert-base-uncased"
MAX_LENGTH = 512


class EmailPreprocessor:
    """
    Handles tokenisation of raw email text for DistilBERT inference.
    Uses the same settings applied during fine-tuning:
    - max_length: 512 tokens
    - padding: max_length
    - truncation: True (tail truncation, preserving header)
    """

    def __init__(self, tokenizer_name: str = TOKENIZER_NAME):
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(
            tokenizer_name
        )
        logger.info("Tokenizer loaded: %s", tokenizer_name)

    def tokenize_single(self, text: str) -> dict:
        """
        Tokenise a single email string for inference.

        Args:
            text: Email text in text_combined format

        Returns:
            Dict with input_ids and attention_mask tensors
        """
        if not text or not text.strip():
            raise ValueError("Input text must not be empty")

        return self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

    @staticmethod
    def build_text_combined(
        sender: str = "",
        date: str = "",
        subject: str = "",
        body: str = "",
    ) -> str:
        """
        Reconstruct text_combined from structured email fields.
        Concatenation order matches the training dataset construction.

        Args:
            sender: Sender email address
            date: Email date string
            subject: Email subject line
            body: Email body text

        Returns:
            Concatenated string in text_combined format
        """
        parts = [p.strip() for p in [sender, date, subject, body]
                 if p.strip()]
        return " ".join(parts)