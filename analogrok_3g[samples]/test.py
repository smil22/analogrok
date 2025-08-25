# ISSA Salim
# June, 18th, 2025
# Model testing file

import torch, string
import components

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = torch.load('./output/model.pth')
model.eval()

n_max = 2
RC = ['!', '?', '.'] # repetition characters
V = list(string.ascii_letters) + [str(i) for i in range(26)] + RC

x = components.encode_token(V, '21', unit = 1) + components.encode_token(V, '?')
x = torch.tensor(x, dtype = torch.long, device = device)
x = x.view(1, -1)
y = model.generate(x, n_max, length = 1)[0].tolist()
ans = components.decode_token(V, y)
print('Prediction: ', ans)