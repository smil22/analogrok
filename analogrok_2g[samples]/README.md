# Dataset

It contains many samples formatted as tuples of input/output pairs.
- Training: [('a!', 'a'), ('b!', 'b'), ... , ('z!', 'z'), ('A?', 'A'), ... , ('Z?', 'Z')]
- Test: [('a?', 'a'), ('b?', 'b'), ... , ('z?', 'z'), ('A!', 'A'), ... , ('Z!', 'Z')]

# Model

- Batch size : 128
- Context length: 2
- Embedding dimension: 128
- Attention heads number: 4
- Attention heads size: 32
- Decoder block number: 1
- Learning rate: 1e-4
