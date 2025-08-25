# Particularity

We pushed the experiment further by adding a third group meant to represent the indexes of the letters, that is, the numbers 
from 0 to 26 (as characters). That implied adding as well a third repetition character and we chose '.'.

# Dataset

It contains many samples formatted as tuples with the first two elements of each one being the input and the last being the output.
- Training: [('a', '!', 'a'), ... , ('z', '!', 'z'), ('A', '?', 'A'), ... , ('Z', '?', 'Z'), ('0', '.', '0'), ... , ('25', '.', '25')]
- Test: [('a', '.', 'a'), ... , ('z', '.', 'z'), ('A', '!', 'A'), ... , ('Z', '!', 'Z'), ('0', '?', '0'), ... , ('25', '?', '25')]

# Model

- Batch size : 128
- Context length: 2
- Embedding dimension: 128
- Attention heads number: 4
- Attention heads size: 32
- Decoder block number: 2
- Learning rate: 1e-4
