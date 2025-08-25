# ISSA Salim
# June, 18th, 2025
# Functions used

import torch
import components
import string, matplotlib.pyplot as plt, numpy as np
from sklearn.manifold import TSNE


def split_V(V, rep_char_number = 3):
    """Splits the vocabulary into lowercases, uppercases and repetition characters groups."""
    
    V_len = len(V)
    limit = int((V_len-rep_char_number) / rep_char_number)
    characters = V[:V_len-rep_char_number]
    rep_chars = V[-rep_char_number:]
    lowercases = characters[:limit]
    remainder = characters[limit:]
    uppercases = remainder[:limit]
    indices = remainder[limit:]
    return lowercases, uppercases, indices, rep_chars


def sample(tpl, V):
    x = torch.tensor(components.encode_token(V, tpl[0], unit = 1) + \
                     components.encode_token(V, tpl[1]), dtype=torch.long)
    y = torch.tensor(components.encode_token(V, tpl[2], unit = 1), dtype=torch.long) 
    return x, y


def get_batch(V, n_max, n_batch, data, device='cpu'):
    """Chooses randomly n_batch samples."""
    
    starting_index = torch.randint(len(data), (n_batch,))
    X, Y = list(), list()
    for i in starting_index:
        tpl = data[i]
        x, y = sample(tpl, V)
        if len(x) != n_max:
            print('Error: context length is different from input_length !')
        X.append(x)
        Y.append(y)
    x = torch.stack(X)
    y = torch.stack(Y)
    x, y = x.to(device), y.to(device)
    return x, y


def generate_data(V):
    """Generates the training and test samples."""
    
    lowercases, uppercases, indices, rep_chars = split_V(V)
    grp_len = len(lowercases)
    
    train_samples, test_samples = [], []
    for i in range(grp_len):
        train_samples.append( ( lowercases[i], rep_chars[0], lowercases[i] ) )
    for i in range(grp_len):
        train_samples.append( ( uppercases[i], rep_chars[1], uppercases[i] ) )
    for i in range(grp_len):
        train_samples.append( ( indices[i], rep_chars[2], indices[i] ) )
    
    for i in range(grp_len):
        test_samples.append( ( lowercases[i], rep_chars[2], lowercases[i] ) )
    for i in range(grp_len):
        test_samples.append( ( uppercases[i], rep_chars[0], uppercases[i] ) )
    for i in range(grp_len):
        test_samples.append( ( indices[i], rep_chars[1], indices[i] ) )
    return train_samples, test_samples


def evaluate(model, data, V, n_max):
    """"Evaluates the model ability to perform well the task."""
    
    model_success = 0
    samples_n = len(data)
    for i in range(samples_n):
        tpl = data[i]
        x, y = sample(tpl, V) 
        x = x.view(1, -1)
        yp = model.generate(x, n_max, length = 1)[0].tolist()
        if y[-1].item() == yp[-1]:
            model_success += 1
    return model_success / samples_n


def tSNE_proj(last_layer_W, V, Id = 0, ax = None):
    """Computes and saves as pictures the t-SNE projection of the last layer weights to visualize
    the embeddings space."""
    
    lowercases, uppercases, indices, rep_chars = split_V(V)
    
    tsne = TSNE(n_components = 2, perplexity = 5, n_iter = 3000, random_state = 42)
    weights_2d = tsne.fit_transform(last_layer_W)
    
    tokens = V if isinstance(V, list) else list(V)
    colors = list()
    for token in tokens:
        if token in lowercases:
            colors.append('midnightblue')
        elif token in uppercases:
            colors.append('darkorange')
        elif token in indices:
            colors.append('darkgreen')
        else:
            colors.append('cyan')
    title = "t-SNE projection of output layer weights"

    x_coords = []
    y_coords = []
    if ax is None: # image saving mode
        plt.figure(figsize=(10, 10))
        for i, token in enumerate(tokens):
            x, y = weights_2d[i]
            x_coords.append(x)
            y_coords.append(y)
            plt.scatter(x, y, color = colors[i])
            plt.annotate(token, (x, y), fontsize=12)
        if Id == 0:
            plt.title(title + " before training")
        elif Id == 1:
            plt.title(title + " after training")
        else:
            plt.title(title)
        #plt.grid(True)
        plt.savefig(f'./output/tSNE_{Id}.jpg', format='jpg', dpi=300)
        plt.close()
        return 0
    else: # animation mode
        ax.set_title('t-SNE Projection of Embeddings', fontsize=10)
        for i, label in enumerate(tokens):
            x, y = weights_2d[i]
            x_coords.append(x)
            y_coords.append(y)
            ax.scatter(x, y, color=colors[i] ,s=10)
            ax.annotate(label, (x, y), fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(True)
        return ax


def W_l2_norm(model):
    """Computes L2 norm of the model parameters."""
    norm = 0.0
    with torch.no_grad():
        for parameter in model.parameters():
            norm += torch.norm(parameter, p = 2) ** 2
    return torch.sqrt(norm)


class Trainer(torch.nn.Module):
    def __init__(self, model, lr, l1_coef, wd, dropout, iters):
        super().__init__()
        self.model = model
        self.lr = lr
        self.wd = wd
        self.iters = iters
        self.l1_coef = l1_coef
        self.dropout_rate = dropout
        
    def train(self, V, n_max, n_batch, train_samples, test_samples, device = 'cpu'):
        """Training loop for the model."""
        
        parameters_number = sum(p.numel() for p in self.model.parameters())
        optimizer = torch.optim.AdamW(self.model.parameters(), lr = self.lr, weight_decay = self.wd)
        
        # Parameters scaling
        # alpha = 1
        # with torch.no_grad():
        #     for param in self.model.parameters():
        #         param *= alpha

        log_obj = open("./output/logs.txt","w")
        with open("./output/logs.txt","a") as log_obj:
            if parameters_number > 1e6:
                log_obj.write("Parameters number: {0:2f} M\n\n".format(parameters_number / 1e6))
            else:
                log_obj.write("Parameters number: {0:2d} \n\n".format(parameters_number))

            # Lists to store training data
            itersLst = np.array([])
            train_accs = np.array([])
            test_accs = np.array([])
            train_loss = np.array([])
            test_losss = np.array([])
            w_norms = np.array([])
            
            tSNE_proj(self.model.lm_head.weight.detach().cpu().numpy(), V, Id = 0) # t-SNE projection
            
            for iter in range(self.iters): # training loop
                print(iter+1)

                xb, yb = get_batch(V, n_max, n_batch, train_samples)
                _, loss = self.model(xb,yb)
                train_loss = np.append(train_loss, loss.item())
                log_obj.write("Step: {0:2d}\tTraining loss: {1:2.2f}\n".format(iter, loss))

                # L1 regularization
                l1_term = torch.tensor(0., requires_grad = True)
                for param in self.model.parameters():
                    l1_term = l1_term + torch.norm(param, 1)

                w_norms = np.append(w_norms, W_l2_norm(self.model).item())
                
                loss = loss + self.l1_coef * l1_term

                self.model.eval()  # sets model to evaluation mode
                with torch.no_grad():  # disables gradient computation
                    xb_test, yb_test = get_batch(V, n_max, n_batch, test_samples)
                    _, test_loss = self.model(xb_test, yb_test)
                    test_losss = np.append(test_losss, test_loss.item())
            
                self.model.train() # restores model to training mode

                # Model's performances evaluation
                itersLst = np.append(itersLst, iter)
                train_acc = evaluate(self.model, train_samples, V, n_max)
                train_accs = np.append(train_accs,train_acc)
                test_acc = evaluate(self.model, test_samples, V, n_max)
                test_accs = np.append(test_accs,test_acc)

                loss.backward()
                optimizer.step()  

        tSNE_proj(self.model.lm_head.weight.detach().cpu().numpy(), V, Id = 1)
        
        torch.save(self.model,"./output/model.pth")

        # Precision plots
        fig1, ax1 = plt.subplots()
        fig1.set_size_inches(12, 6)
        ax1.plot(itersLst, train_accs, color = 'blue', label = 'Train acc.')
        ax1.plot(itersLst, test_accs, color = 'orange', label = 'Test acc.')
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Precision")
        ax1.legend(loc="lower right")
        ax1.grid()

        ax2 = ax1.twinx()
        ax2.semilogy(itersLst, w_norms, color = 'green', label = "W norm")
        ax2.set_ylabel("Weights L2 norm")
        ax2.legend(loc="upper left")
        plt.title('Model precision and weights norm growth')
        plt.savefig('./output/precision_plot.jpg', format='jpg', dpi=300)
        plt.close()

        # Losses plots
        fig2, ax3 = plt.subplots()
        fig2.set_size_inches(12, 6)
        ax3.semilogy(itersLst, train_loss, color = 'blue', label = 'Training loss')
        ax3.semilogy(itersLst, test_losss, color = 'orange', label = 'Test loss')
        ax3.set_xlabel("Epochs")
        ax3.set_ylabel("Losses")
        ax3.legend(loc="best")
        ax3.grid()
        plt.title('Model losses')
        plt.savefig('./output/loss_plot.jpg', format='jpg', dpi=300)
        plt.close()