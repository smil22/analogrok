# ISSA Salim
# June, 18th, 2025
# Functions used

import torch
import torch.nn.functional as F
import components
import matplotlib.pyplot as plt, numpy as np, matplotlib.patches as mpatches, \
    matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE


def split_V(V, rep_char_number = 2):
    """Splits the vocabulary into lowercases, uppercases and repetition characters groups."""
    
    V_len = len(V)
    limit = int((V_len-rep_char_number) / rep_char_number)
    characters = V[:V_len-rep_char_number]
    rep_chars = V[-rep_char_number:]
    lowercases = characters[:limit]
    uppercases = characters[limit:]
    return lowercases, uppercases, rep_chars


def generate_data(V, rule_factor = 1):
    """Generates the training and test texts."""
    
    lowercases, uppercases, rep_chars = split_V(V)
    
    train_text, test_text = '', ''
    
    # Training text
    for char in lowercases:
        train_text += (char + rep_chars[0] + char) * rule_factor
    for char in uppercases:
        train_text += (char + rep_chars[1] + char) * rule_factor
        
    # Test text
    for char in lowercases:
        test_text += (char + rep_chars[1] + char) * rule_factor
    for char in uppercases:
        test_text += (char + rep_chars[0] + char) * rule_factor
        
    return train_text, test_text 


def get_batch(n_max, n_batch, encoded_text, device = 'cpu'):
    """Picks randomly n_batch data samples of n_max length: (n_max, d_model, n_batch)."""
    
    starting_index = torch.randint(len(encoded_text)-n_max, (n_batch,))
    x = torch.stack([encoded_text[i:i+n_max] for i in starting_index])
    y = torch.stack([encoded_text[i+1:i+n_max+1] for i in starting_index])
    x, y = x.to(device), y.to(device)
    return x, y


def evaluate(model, encoded_text, V, n_max):
    """"Evaluates the model ability to perform well the task."""

    _, _, rep_chars = split_V(V)
    encoded_rc = components.encode_token(V, ''.join(rep_chars))
    encoded_rc = torch.tensor(encoded_rc, device = model.embTab.weight.device)
    
    rule_occurences = 0
    model_success = 0
    text_len = len(encoded_text)
    for i in range(text_len - n_max):
        if encoded_text[i:i+n_max][-1].tolist() == encoded_rc[0].item() or \
            encoded_text[i:i+n_max][-1].tolist() == encoded_rc[1].item():
            rule_occurences += 1
            x = encoded_text[i:i+n_max]
            y = x[-2].item()
            x = x.view(1, -1)
            yp = model.generate(x, n_max, length = 1)[0].tolist()
            if yp[-1] == y:
                model_success += 1
    score = model_success / rule_occurences
    return score


class tSNE_proj:
    """Performs tSNE projection of the last layer/output layer weights of the model across epochs."""
    
    def __init__(self, model, stacked_weights, rc_distance, V, iters):
        
        self.model = model
        self.stacked_weights = stacked_weights # (iters * V_len, d_model)
        self.V = V
        self.iters = iters
        self.rc_distance = rc_distance
        
        tsne = TSNE(n_components = 2, perplexity = 5, n_iter = 3000, random_state = 42)
        embedded_2d = tsne.fit_transform(self.stacked_weights)
        self.proj_by_epoch = np.split(embedded_2d, self.iters)  # list of (V_len, n_components)
        
    def process(self, epoch, ax = None):
        """Draws the projection for a particular epoch and saves it as an image or returns it on the
        given ax."""
        
        lowercases, uppercases, rep_chars = split_V(self.V)
        tokens = self.V if isinstance(self.V, list) else list(self.V)
        colors = list()
        for token in tokens: # groups colors setting
            if token in lowercases:
                colors.append('midnightblue')
            elif token in uppercases:
                colors.append('darkorange')
            else:
                colors.append('cyan')
        
        dots_cloud = self.proj_by_epoch[epoch]
        
        if ax is None:
            fig, _ = plt.subplots(figsize = (10, 10)) # ax setting
            plt.title(f't-SNE projection of embeddings at epoch {epoch + 1}', fontsize=15)
    
            x_coords = []
            y_coords = []
            for i, label in enumerate(tokens):
                x, y = dots_cloud[i]
                x_coords.append(x)
                y_coords.append(y)
                plt.scatter(x, y, color=colors[i] , s = 15)
                plt.annotate(label, (x, y), fontsize = 12)
            
            fig.text(0.5, 0.01, f"Cosine distance: {self.rc_distance[epoch]:.5e}", 
                     ha='center', va='bottom', fontsize=15)
            plt.xticks([])
            plt.yticks([])
            
            plt.savefig(f'./output/tSNE_{epoch + 1}.jpg', format='jpg', dpi=300)
            plt.close()
            return 0
        else:
            ax.set_title('t-SNE projection of embeddings', fontsize=15)
            rc_distanceTxt = ax.text(
                0.5, -0.1, "",
                transform=ax.transAxes,
                fontsize=11,
                ha='center',
                va='bottom',
                color='black',
                label = 'subtitle'
            )
    
            x_coords = []
            y_coords = []
            for i, label in enumerate(tokens):
                x, y = dots_cloud[i]
                x_coords.append(x)
                y_coords.append(y)
                ax.scatter(x, y, color=colors[i] , s = 15)
                ax.annotate(label, (x, y), fontsize = 12)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(True)
            rc_distanceTxt.set_text(f"Cosine distance: {self.rc_distance[epoch]:.5e}")
            
            return ax
    
    
def rep_chars_distance(V, model):
    """Computes the cosine distance between both repetition character embeddings."""
    
    _, _, rep_chars = split_V(V)
    encoded_rc = components.encode_token(V, ''.join(rep_chars))
    encoded_rc = torch.tensor(encoded_rc, device=model.embTab.weight.device)
    rc_embeddings = model.embTab(encoded_rc)  # shape: [2, embedding_dim]

    similarity = F.cosine_similarity(rc_embeddings[0].unsqueeze(0),
                                     rc_embeddings[1].unsqueeze(0), dim=1)
    
    # cosine distance
    distance = 1 - similarity
    
    return distance


def W_l2_norm(model):
    """Computes L2 norm of the model parameters."""
    norm = 0.0
    with torch.no_grad():
        for parameter in model.parameters():
            norm += torch.norm(parameter, p = 2) ** 2
    return torch.sqrt(norm)


def update_anim(epoch, axes, lowercasesPdct, uppercasesPdct, lowercasesTxt, uppercasesTxt, group_len, V, 
                lowercasesImg, uppercasesImg, train_acc, test_acc, train_accTxt, 
                test_accTxt, rep_char_number, weights, dim_reducer):
    """Updates the animation for each epoch."""
    
    lowercases, uppercases, rep_chars = split_V(V)
    
    axes[2].clear()
    
    lowercasesTab = lowercasesPdct[epoch] # lowercases predictions table for this epoch
    uppercasesTab = uppercasesPdct[epoch] # uppercases predictions table for this epoch
    weights = weights[epoch] # last layer weights for this epoch
    lowercasesBox = np.zeros((group_len, rep_char_number))  # 0: No prediction, 1: Correct, -1: Incorrect
    uppercasesBox = np.zeros((group_len, rep_char_number))
    
    # Tables filling and boxes coloring
    for i in range(group_len):
        for j in range(rep_char_number):
            lowercasePdct = lowercasesTab[i, j]
            uppercasePdct = uppercasesTab[i, j]
            correct_lowercase = (lowercasePdct == lowercases[i])
            correct_uppercase = (uppercasePdct == uppercases[i])
            lowercasesBox[i, j] = 1 if correct_lowercase else -1  
            uppercasesBox[i, j] = 1 if correct_uppercase else -1
            lowercasesTxt[i, j].set_text(lowercasePdct)
            uppercasesTxt[i, j].set_text(uppercasePdct)
            lowercasesTxt[i, j].set_color("black")
            uppercasesTxt[i, j].set_color("black")
    
    lowercasesImg.set_array(lowercasesBox)
    uppercasesImg.set_array(uppercasesBox)

    axes[0].figure.suptitle(f"Epoch {epoch} \n", fontsize = 16)
    
    # t-SNE projection
    axes[2] = dim_reducer.process(epoch = epoch, ax = axes[2])
    
    # Accuracies update
    train_accTxt.set_text(f"Train Acc: {train_acc[epoch]:.2f}%")
    test_accTxt.set_text(f"Test Acc: {test_acc[epoch]:.2f}%")
    #rc_distanceTxt.set_text(f"Cosine distance: {rc_distance[epoch]:.2f}")
    
    return [lowercasesImg, uppercasesImg, train_accTxt, test_accTxt, axes[2]]


def save_anim(lowercasesPdct, uppercasesPdct, V, iters, train_acc, test_acc, weights, dim_reducer):

    lowercases, uppercases, rep_chars = split_V(V)
    group_len = len(lowercases)
    rep_char_number = len(rep_chars)
    
    lowercasesBox = np.zeros((group_len, rep_char_number))  # 0: No prediction, 1: Correct, -1: Incorrect
    uppercasesBox = np.zeros((group_len, rep_char_number))  # Uppercases boxes colors

    # Colormap for the predictions (white=0, green=1, red=-1)
    cmap = ListedColormap(["red", "white", "green"])

    fig, axes = plt.subplots(nrows=1,ncols=3,figsize=(12, 6))
    lowercasesImg = axes[0].imshow(lowercasesBox, cmap=cmap, vmin=-1, vmax=1)
    uppercasesImg = axes[1].imshow(uppercasesBox, cmap=cmap, vmin=-1, vmax=1)
    
    axes[0].set_xticks(np.arange(rep_char_number) + 0.5)  
    axes[0].set_yticks(np.arange(group_len) + 0.5)
    axes[1].set_xticks(np.arange(rep_char_number) + 0.5)
    axes[1].set_yticks(np.arange(group_len) + 0.5)
    axes[0].xaxis.set_label_position('top')
    axes[0].xaxis.set_ticks_position('top')
    axes[1].xaxis.set_label_position('top')
    axes[1].xaxis.set_ticks_position('top')
    axes[0].tick_params(axis='x', which='both', bottom=False, top=False)
    axes[0].tick_params(axis='y', which='both', left=False, right=False)
    axes[1].tick_params(axis='x', which='both', bottom=False, top=False)
    axes[1].tick_params(axis='y', which='both', left=False, right=False)
    axes[0].set_xticklabels(rep_chars, fontsize=12, horizontalalignment='center', verticalalignment='bottom')
    axes[0].set_yticklabels(lowercases, fontsize=12, horizontalalignment='right', verticalalignment='center')
    axes[1].set_xticklabels(rep_chars, fontsize=12, horizontalalignment='center', verticalalignment='bottom')
    axes[1].set_yticklabels(uppercases, fontsize=12, horizontalalignment='right', verticalalignment='center')
    for tick in axes[0].xaxis.get_majorticklabels():
        tick.set_horizontalalignment("right")
    for tick in axes[1].xaxis.get_majorticklabels():
        tick.set_horizontalalignment("right")
    for tick in axes[0].yaxis.get_majorticklabels():
        tick.set_verticalalignment("bottom")     
    for tick in axes[1].yaxis.get_majorticklabels():
        tick.set_verticalalignment("bottom")
    axes[0].grid(True, which='both', axis='both', color='black', linestyle='-', linewidth=1,
            alpha=0.5, zorder=3)
    axes[1].grid(True, which='both', axis='both', color='black', linestyle='-', linewidth=1,
            alpha=0.5, zorder=3)
    
    correct_patch = mpatches.Patch(color='green', label='Correct')
    incorrect_patch = mpatches.Patch(color='red', label='Incorrect')
    axes[0].legend(handles=[correct_patch, incorrect_patch], 
          loc='upper right', 
          bbox_to_anchor=(-3, 1),  # Moves legend outside the plot (right side)
          fontsize=10, 
          frameon=False)
    
    train_accTxt = axes[0].text(
    -7, 0.75, "",
    transform=axes[0].transAxes,
    fontsize=11,
    color="blue"
    )
    test_accTxt = axes[0].text(
        -7, 0.7, "",
        transform=axes[0].transAxes,
        fontsize=11,
        color="darkgoldenrod"
    )
    test_accTxt.set_fontweight('bold')
    train_accTxt.set_fontweight('bold')
    
    lowercasesTxt = np.empty((group_len, rep_char_number), dtype=object)
    uppercasesTxt = np.empty((group_len, rep_char_number), dtype=object)
    for i in range(group_len):
        for j in range(rep_char_number):
            lowercasesTxt[i, j] = axes[0].text(j, i, '', ha='center', va='center', fontsize=12, color='black', zorder=4)
            uppercasesTxt[i, j] = axes[1].text(j, i, '', ha='center', va='center', fontsize=12, color='black', zorder=4)
    
    dot_chars = ['●', '●']
    dot_colors = ['blue', 'gold']
    for j, (dot, color) in enumerate(zip(dot_chars, dot_colors)):
        axes[0].text(
            j,
            group_len + 1.5,
            dot,
            ha='center',
            va='bottom',
            fontsize=15,
            color=color,
            zorder=5
        )  
    for j, (dot, color) in enumerate(zip(dot_chars, dot_colors[::-1])):
        axes[1].text(
            j,
            group_len + 1.5,
            dot,
            ha='center',
            va='bottom',
            fontsize=15,
            color=color,
            zorder=5
        )
    
    ani = animation.FuncAnimation(fig, update_anim, frames = iters, 
                              fargs=(axes, lowercasesPdct, uppercasesPdct, lowercasesTxt, uppercasesTxt, group_len, V, 
                                              lowercasesImg, uppercasesImg, train_acc, test_acc, train_accTxt, 
                                              test_accTxt, rep_char_number, weights, dim_reducer), interval = 1000)
    ani.save('./output/training_table_animation.mp4',
             writer=animation.FFMpegWriter(fps=1, bitrate=5000, codec='mpeg4'),
             dpi=300)
    
    plt.close(fig)
    
    return 0

class Trainer(torch.nn.Module):
    def __init__(self, model, lr, l1_coef, wd, dropout, iters, anim = 0):
        super().__init__()
        self.model = model
        self.lr = lr
        self.wd = wd
        self.iters = iters
        self.l1_coef = l1_coef
        self.dropout_rate = dropout
        self.anim = anim
        self.weights = []
        self.rc_distance = []
        if self.anim:
            self.lowercasesPdct, self.uppercasesPdct = [], []
        
    def anim_data(self, model, V, n_max, device = 'cpu'):
        """Gathers all the data needed to process the animation."""
        
        lowercases, uppercases, rep_chars = split_V(V)
        group_len = len(lowercases)
        rep_char_number = len(rep_chars)
        
        lowercasesPdctTab = np.full((group_len, rep_char_number), '', dtype='U1')
        uppercasesPdctTab = np.full((group_len, rep_char_number), '', dtype='U1')
        
        for i in range(group_len): # model evaluation for all entries at this epoch
            for j in range(rep_char_number):
                lowercase_x = components.encode_token(V, lowercases[i] + rep_chars[j])  # Get prediction
                uppercase_x = components.encode_token(V, uppercases[i] + rep_chars[j])  # Get prediction
                lowercase_x = torch.tensor(lowercase_x, dtype=torch.long, device=device)
                uppercase_x = torch.tensor(uppercase_x, dtype=torch.long, device=device)
                lowercase_x = lowercase_x.view(1, -1)
                uppercase_x = uppercase_x.view(1, -1)
                lowercase_y = model.generate(lowercase_x, n_max, length= 1)[0].tolist()
                uppercase_y = model.generate(uppercase_x, n_max, length= 1)[0].tolist()
                l_predicted_letter = components.decode_token(V, lowercase_y)[-1]
                u_predicted_letter = components.decode_token(V, uppercase_y)[-1]
                lowercasesPdctTab[i, j] = l_predicted_letter  # Store in table
                uppercasesPdctTab[i, j] = u_predicted_letter
                
        self.lowercasesPdct.append(lowercasesPdctTab.copy())
        self.uppercasesPdct.append(uppercasesPdctTab.copy())
        self.weights.append(model.lm_head.weight.detach().cpu().numpy().copy())
        self.rc_distance.append(rep_chars_distance(V, model).item())
        
    def train(self, V, n_max, n_batch, train_text, test_text, device = 'cpu'):
        """Training loop for the model."""
        
        lowercases, uppercases, rep_chars = split_V(V)
        
        training_data = torch.tensor(components.encode_token(V, train_text), dtype = torch.long)
        test_data = torch.tensor(components.encode_token(V, test_text), dtype = torch.long)
        
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
            
            for iter in range(self.iters): # training loop
                print(iter+1)

                xb, yb = get_batch(n_max, n_batch, training_data)
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
                    xb_test, yb_test = get_batch(n_max, n_batch, test_data)
                    _, test_loss = self.model(xb_test, yb_test)
                    test_losss = np.append(test_losss, test_loss.item())
            
                self.model.train() # restores model to training mode

                # Model's performances evaluation
                itersLst = np.append(itersLst, iter)
                train_acc = evaluate(self.model, training_data, V, n_max)
                train_accs = np.append(train_accs,train_acc)
                test_acc = evaluate(self.model, test_data, V, n_max)
                test_accs = np.append(test_accs,test_acc)
                
                self.weights.append(self.model.lm_head.weight.detach().cpu().numpy().copy())
                self.rc_distance.append(rep_chars_distance(V, self.model).item())
                
                if self.anim:
                    self.anim_data(self.model, V, n_max)

                loss.backward()
                optimizer.step()  

        stacked_weights = np.vstack(self.weights)  # (iters * V_len, d_model)
        dim_reducer = tSNE_proj(self.model, stacked_weights, self.rc_distance, V, self.iters)
        dim_reducer.process(epoch = 0)
        dim_reducer.process(epoch = self.iters - 1)
        
        torch.save(self.model,"./output/model.pth")

        if self.anim:
            save_anim(self.lowercasesPdct, self.uppercasesPdct, V, self.iters, train_accs * 100, 
                      test_accs * 100, self.weights, dim_reducer)

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