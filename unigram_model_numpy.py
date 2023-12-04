"""Demonstrate the "unigram" language model."""
from collections import Counter
import math
from typing import Optional, List
import numpy as np
from numpy.typing import NDArray

# generate English document
text = "Four score and seven years ago, our fathers brought forth, upon this continent, a new nation, conceived in liberty, and dedicated to the proposition that all men are created equal."
text = (text + " ") * 100  # make the document very long

# tokenize - split the document into a list of little strings
tokens = [char for char in text]

# encode as {0, 1}
vocabulary = [chr(i + ord("a")) for i in range(26)] + [" ", None]

FloatArray = NDArray[np.float64]


def onehot(
    vocabulary: List[Optional[str]], token: Optional[str]
) -> FloatArray:
    """Generate the one-hot encoding for the provided token in the provided vocabulary."""
    embedding = np.zeros((len(vocabulary), 1))
    try:
        idx = vocabulary.index(token)
    except ValueError:
        idx = len(vocabulary) - 1
    embedding[idx, 0] = 1
    return embedding


encodings = [onehot(vocabulary, token) for token in tokens]


# define model
class UnigramModel:
    """The unigram language model."""

    def __init__(self) -> None:
        """Initialize."""
        self.p: Optional[FloatArray] = None

    def train(self, encodings: List[FloatArray]) -> "UnigramModel":
        """Train the model on data."""
        counts = np.ones((len(vocabulary), 1))
        for encoding in encodings:
            counts += encoding
        self.p = counts / counts.sum()
        return self

    def apply(self, encodings: List[FloatArray]) -> float:
        """Compute the log probability of a document."""
        if self.p is None:
            raise ValueError("This model is untrained")
        return sum(math.log(encoding.T @ self.p) for encoding in encodings)


# train model
model = UnigramModel()
model.train(encodings)

# compute probability
test_data = "The quick brown fox jumps over the lazy dog."
tokens = [char for char in test_data]
encodings = [onehot(vocabulary, token) for token in tokens]
log_p = model.apply(encodings)

# print
print(f"learned p value: {model.p}")
print(f"log probability of document: {log_p}")
print(f"probability of document: {math.exp(log_p)}")
