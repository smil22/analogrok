# ISSA Salim
# June, 18th, 2025
# Model components parts

import torch


def encode_token(V, string, unit = 0):
    """"Encodes tokens with their indices in vocabulary."""
    stoi = {ch:i for i,ch in enumerate(V)}
    if unit:
        return [stoi[string]]
    else:
        return [stoi[char] for char in string]


def decode_token(V, indices, unit = 0):
    """"Decodes tokens according to their indices in vocabulary."""
    itos = {i:ch for i,ch in enumerate(V)}
    if unit:
        return ''.join(itos[indices])
    else:
        return ''.join([itos[integer] for integer in indices])


class AttentionHead(torch.nn.Module):
    """One head of self-attention."""
    
    def __init__(self, dk, d_model, n_max, dropout):
        super().__init__()
        self.wK = torch.nn.Linear(d_model, dk, bias = False)
        self.wQ = torch.nn.Linear(d_model, dk, bias = False)
        self.wV = torch.nn.Linear(d_model, dk, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(n_max, n_max)))
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, x):
        n_batch, n_max, d_model = x.shape
        Q = self.wQ(x)
        K = self.wK(x)
        
        # Attention scores
        P = Q @ K.transpose(-2,-1) * d_model**-0.5 # (n_batch,n_max,d_model) @ (n_batch,d_model,n_max) -> (n_batch,n_max,n_max)
        P = P.masked_fill(self.tril[:n_max,:n_max] == 0, float('-inf')) # (n_batch,n_max,n_max)
        P = torch.nn.functional.softmax(P, dim=-1) # (n_batch,n_max,n_max)
        P = self.dropout(P)
        V = self.wV(x) # (n_batch,n_max,d_model)
        O = P @ V # (n_batch,n_max,n_max) @ (n_batch,n_max,d_model) -> (n_batch,n_max,d_model)
        return O


class MultiHeadAttention(torch.nn.Module):
    """Mutilple heads of self-attention in parallel."""
    
    def __init__(self, h, dk, d_model, n_max, dropout):
        super().__init__()
        self.heads = torch.nn.ModuleList([AttentionHead(dk, d_model, n_max, dropout) for _ in range(h)])
        self.proj = torch.nn.Linear(h * dk, d_model)
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self,x):
        O = torch.cat([h(x) for h in self.heads], dim=-1) # (n_batch,n_max,h*dk)
        O = self.dropout(self.proj(O)) # (n_batch,n_max,d_model)
        return O
    

class FeedForward(torch.nn.Module):
    """"A simple linear layer followed by a non-linearity."""
    
    def __init__(self, d_model, dropout):
        super().__init__()
        output_scale = 4
        self.network = torch.nn.Sequential(
            torch.nn.Linear(d_model, output_scale*d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(output_scale*d_model,d_model), # projection layer
            torch.nn.Dropout(dropout)
        )
        
    def forward(self,x):
        return self.network(x)


class DecoderBlock(torch.nn.Module):
    """The decoder."""
    
    def __init__(self, h, dk, d_model, dropout, n_max):
        super().__init__()
        self.sa = MultiHeadAttention(h, dk, d_model, n_max, dropout)
        self.ffwd = FeedForward(d_model, dropout)
        self.ln1 = torch.nn.LayerNorm(d_model)
        self.ln2 = torch.nn.LayerNorm(d_model)
        
    def forward(self,x):
        x = x + self.sa(self.ln1(x)) # residual connection
        x = x + self.ffwd(self.ln2(x))
        return x
    

class SLM(torch.nn.Module):
    """The transformer model."""
    
    def __init__(self, V_len, d_model, n_max, h, N, dk, dropout):
        super().__init__()
        self.embTab = torch.nn.Embedding(V_len, d_model)
        self.posEmbTab = torch.nn.Embedding(n_max, d_model)
        self.blocks = torch.nn.Sequential(*[DecoderBlock(h, dk, d_model, dropout, n_max) for _ in range(N)])
        self.ln_f = torch.nn.LayerNorm(d_model)
        self.lm_head = torch.nn.Linear(d_model, V_len)
        self.d_model = d_model
        self.h = h
        self.N = N
        
    def forward(self, index, targets=None):
        n_batch, n_max = index.shape
        index = index.to(self.embTab.weight.device)
        tok_emb = self.embTab(index) # (n_batch, n_max, d_model)
        pos_emb = self.posEmbTab(torch.arange(n_max, device=index.device)) # (n_max, d_model)
        x = tok_emb + pos_emb # (n_batch, n_max, d_model)
        x = self.blocks(x) # (n_batch, n_max, d_model)
        x = self.ln_f(x) # (n_batch, n_max, d_model)
        logits = self.lm_head(x) # (n_batch, n_max, V_len)
        
        if targets is None:
            loss = None
        else:
            n_batch, n_max, d_model = logits.shape
            logits = logits[:,-1,:] # (n_batch, d_model)
            targets = targets.view(n_batch)
            loss = torch.nn.functional.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, index, n_max, length = 500, T = 1):
        for _ in range(length):
            # crop index to the last n_max tokens
            index_cond = index[:,-n_max:]
            logits, _ = self(index_cond)
            # focus only on the last time step
            logits = logits[:,-1,:] # (n_batch, d_model)
            index_next = torch.argmax(logits / T, dim=-1, keepdim = True)
            # softmax applying to get probabilities
            #probs = torch.nn.functional.softmax(logits / T, dim=-1) # (n_batch,d_model)
            # sample from the distribution
            #index_next = torch.multinomial(probs,num_samples=1) #(n_batch,1)
            # append sampled index to the running sequence
            device = index.device
            index_next = index_next.to(device)
            index = torch.cat((index,index_next), dim = 1) # (n_batch, n_max+1)
        return index