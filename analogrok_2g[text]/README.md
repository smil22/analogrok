# Particularity
That is the case resembling the most to the classical processing environment of a language model. Indeed we trained the model on data structured
as raw text. We discovered that the model can always achieve the task but in a particular way, that is, 
by [in-context learning](https://www.lakera.ai/blog/what-is-in-context-learning) and more precisely by one-shot learning. 
You can test it with the **test.py** file.

# Dataset

- Training: 'a!ab!b...y!yz!zA?AB?B...Y?YZ?Z'
- Test: 'a?ab?b...y?yz?zA!AB!B...Y!YZ!Z'

# Model

- Batch size : 128
- Context length: 8
- Embedding dimension: 128
- Attention heads number: 4
- Attention heads size: 32
- Decoder block number: 1
- Learning rate: 1e-4
