# ISSA Salim
# June, 18th, 2025
# This is the main file to execute.

import torch
import components, tools
import string


iters = 1000 # training iterations number

# Hyperparameters of the model
n_batch = 128 # number of elements in a batch (n_max, d_model)
n_max = 8 # context length
d_model = 128 # tokens embedding dimension
h = 4   # number of attention heads
dk = 32 # attention head size
N = 1 # number of decoder block

# Adam optimizer tuning
lr = 1e-4 # learning rate for the optimizer
wd = 0 # L2 regularization term / weight decay
l1_coef = 0 # L1 regularization term
#betas=(0.8, 0.999)

dropout = 0

device = 'cuda' if torch.cuda.is_available() else 'cpu'

anim = 0 # model learning record activation parameter

torch.manual_seed(0)

# Data generation
RC = ['!', '?'] # repetition characters
V = list(string.ascii_letters) + RC
V_len = len(V)

train_text, test_text = tools.generate_data(V)

# Model creation
model = components.SLM(V_len, d_model, n_max, h, N, dk, dropout) 
model = model.to(device) # hardware for the model to be processed on (CPU or GPU)

# Training
trainer = tools.Trainer(model, lr, l1_coef, wd, dropout, iters, anim)
trainer.train(V, n_max, n_batch, train_text, test_text)