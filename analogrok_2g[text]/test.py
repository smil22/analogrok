# ISSA Salim
# June, 18th, 2025
# Model testing file

import torch, string
import tools, components

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = torch.load('./output/model.pth')
model.eval()

n_max = 8
RC = ['!', '?'] # repetition characters
V = list(string.ascii_letters) + RC

x = components.encode_token(V, 'a!ab?')
x = torch.tensor(x, dtype = torch.long, device = device)
x = x.view(1, -1)
y = model.generate(x, n_max, length = 1)[0].tolist()
ans = components.decode_token(V, y)
print('Prediction: ', ans)

distance = tools.rep_chars_distance(V, model)
print('Repetition characters distance: {0:2.2e}'.format(distance))