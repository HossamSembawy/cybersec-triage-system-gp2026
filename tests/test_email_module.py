from unittest.mock import patch

import pytest
from transformers import DistilBertTokenizerFast

from src.email_module.preprocessor import EmailPreprocessor


@pytest.mark.parametrize("text", ["", "   ", "\t\n"])
def test_empty_email_is_rejected_before_tokenization(text):
    # Avoid downloading the tokenizer because validation happens first.
    with patch.object(
        DistilBertTokenizerFast,
        "from_pretrained",
        return_value=None,
    ):
        preprocessor = EmailPreprocessor()

    with pytest.raises(ValueError, match="must not be empty"):
        preprocessor.tokenize_single(text)
