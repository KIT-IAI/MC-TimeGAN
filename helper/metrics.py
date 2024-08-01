"""metrics.py"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch import tensor
import numpy as np
from itertools import chain
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Determine the device to use for computation
device = (
    'cuda'
    if torch.cuda.is_available()
    else 'mps'
    if torch.backends.mps.is_available()
    else 'cpu'
)

class discriminator(nn.Module):
    def __init__(self, input_features, hidden_dim, epochs, batch_size):
        super().__init__()
        self.input_features = input_features
        self.hidden_dim = hidden_dim
        self.num_layers = 1
        self.epochs = epochs
        self.batch_size = batch_size
        self.rnn = nn.GRU(input_size=input_features, hidden_size=hidden_dim, num_layers=self.num_layers, batch_first=True)
        self.model = nn.Linear(hidden_dim, 1)
        self.activation = nn.Sigmoid()
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(chain(self.rnn.parameters(), self.model.parameters()))

    def forward(self, x):
        _, d_last_states = self.rnn(x)
        y_hat_logit = self.model(torch.swapaxes(d_last_states, 0, 1))
        y_hat = self.activation(y_hat_logit)
        return y_hat_logit, y_hat

    def fit(self, x, x_hat):
        x_train, x_test, x_hat_train, x_hat_test = train_test_split(x, x_hat, test_size=0.2)
        x_train, x_hat_train = tensor(x_train, dtype=torch.float32, device=device), tensor(x_hat_train, dtype=torch.float32, device=device)
        dataset_train = TensorDataset(x_train, x_hat_train)

        x_test, x_hat_test = tensor(x_test, dtype=torch.float32, device=device, requires_grad=False), tensor(x_hat_test, dtype=torch.float32, device=device, requires_grad=False)
        for itt in tqdm(range(self.epochs)):
            batches = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True)

            self.train()
            for x, x_hat in batches:
                self.optimizer.zero_grad()
                y_logit_real, y_pred_real = self.forward(x)
                y_logit_fake, y_pred_fake = self.forward(x_hat)
                d_loss_real = torch.mean(self.loss_fn(y_logit_real, torch.ones_like(y_logit_real, dtype=torch.float32, device=device, requires_grad=False)))
                d_loss_fake = torch.mean(self.loss_fn(y_logit_fake, torch.zeros_like(y_logit_fake, dtype=torch.float32, device=device, requires_grad=False)))
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                self.optimizer.step()
        
        self.eval()
        with torch.no_grad():
            _, y_pred_real = self.forward(x_test)
            _, y_pred_fake = self.forward(x_hat_test)
            y_pred_final = np.squeeze(np.concatenate((y_pred_real.cpu().detach().numpy(), y_pred_fake.cpu().detach().numpy()), axis=0))
            y_label_final = np.concatenate((np.ones([len(y_pred_real,)]), np.zeros([len(y_pred_fake,)])), axis=0)
            acc = accuracy_score(y_label_final, (y_pred_final > 0.5))
            discriminative_score = np.abs(0.5 - acc)

        return discriminative_score

def discriminative_score_metrics(ori_data, generated_data):
    no, seq_len, dim = ori_data.shape
    hidden_dim = int(dim / 2)
    if hidden_dim == 0:
        hidden_dim = 1
    iterations = 2000
    batch_size = 128

    print(f'Using {device} device')
    model = discriminator(input_features=dim, hidden_dim=hidden_dim, epochs=iterations, batch_size=batch_size).to(device)
    discriminative_score = model.fit(ori_data, generated_data)

    return discriminative_score

class predictor(nn.Module):
    def __init__(self, dim, hidden_dim, epochs, batch_size):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_layers = 1
        self.epochs = epochs
        self.batch_size = batch_size
        self.rnn = nn.GRU(input_size=(dim-1) if dim > 1 else 1, hidden_size=hidden_dim, num_layers=self.num_layers, batch_first=True)
        self.model = nn.Sequential(nn.Linear(self.hidden_dim, 1), nn.Sigmoid())
        self.loss_fn = nn.L1Loss()
        self.optimizer = torch.optim.Adam(chain(self.rnn.parameters(), self.model.parameters()), 1e-3)

    def forward(self, x):
        p_outputs, _ = self.rnn(x)
        return self.model(p_outputs)

    def fit(self, data_train, data_test):
        if self.dim == 1:
            x_train, y_train = tensor(data_train[:,:-1,:], dtype=torch.float32, device=device), tensor(data_train[:,1:,:], dtype=torch.float32, device=device)
        elif self.dim > 1:
            x_train = data_train[:,:-1,:(self.dim-1)]
            y_train = np.reshape(data_train[:,1:,(self.dim-1)], (data_train.shape[0], data_train.shape[1]-1, 1))
            x_train, y_train = tensor(x_train, dtype=torch.float32, device=device), tensor(y_train, dtype=torch.float32, device=device)

        dataset = TensorDataset(x_train, y_train)
        for itt in tqdm(range(self.epochs)):
            batches_train = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            self.train()
            for X, Y in batches_train:
                self.optimizer.zero_grad()
                pred_train = self.forward(X)
                loss = self.loss_fn(Y, pred_train)
                loss.backward()
                self.optimizer.step()

        if self.dim == 1:
            x_test = tensor(data_test[torch.randperm(len(data_test), device=device).cpu().detach().numpy()][:-1,:], dtype=torch.float32, device=device, requires_grad=False)
            y_test = tensor(data_test[torch.randperm(len(data_test), device=device).cpu().detach().numpy()][1:,:], dtype=torch.float32, device=device, requires_grad=False)
        elif self.dim > 1:
            x_test = data_test[:,:-1,:(self.dim-1)]
            y_test = np.reshape(data_test[:,1:,(self.dim-1)], (data_test.shape[0], data_test.shape[1]-1, 1))
            x_test, y_test = tensor(x_test, dtype=torch.float32, device=device, requires_grad=False), tensor(y_test, dtype=torch.float32, device=device, requires_grad=False)

        MAE = 0
        self.eval()
        with torch.no_grad():
            pred_test = self.forward(x_test)
            for i in range(len(pred_test)):
                MAE += mean_absolute_error(y_test[i,:,:].cpu().detach().numpy(), pred_test[i,:,:].cpu().detach().numpy())
                break

        return MAE

def predictive_score_metrics(ori_data, generated_data):
    no, seq_len, dim = np.asarray(ori_data).shape
    hidden_dim = int(dim / 2)
    if hidden_dim == 0:
        hidden_dim = 1
    iterations = 5000
    batch_size = 128

    print(f'Using {device} device')

    model = predictor(dim, hidden_dim, iterations, batch_size).to(device)
    predictive_score = model.fit(generated_data, ori_data)
    return predictive_score / no

def visualization(ori_data, generated_data, analysis):
    anal_sample_no = min(1000, len(ori_data))
    idx = np.random.permutation(len(ori_data))[:anal_sample_no]
    ori_data = np.asarray(ori_data)[idx]
    generated_data = np.asarray(generated_data)[idx]
    no, seq_len, dim = ori_data.shape
    prep_data = np.reshape(np.mean(ori_data, axis=2), (anal_sample_no, seq_len))
    prep_data_hat = np.reshape(np.mean(generated_data, axis=2), (anal_sample_no, seq_len))
    colors = ["red"] * anal_sample_no + ["blue"] * anal_sample_no
    if analysis == 'pca':
        pca = PCA(n_components=2)
        pca_results = pca.fit_transform(prep_data)
        pca_hat_results = pca.transform(prep_data_hat)
        _plot_results(pca_results, pca_hat_results, colors, "PCA plot", "x-pca", "y-pca")
    elif analysis == 'tsne':
        prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(prep_data_final)
        _plot_results(tsne_results[:anal_sample_no], tsne_results[anal_sample_no:], colors, "t-SNE plot", "x-tsne", "y-tsne")

def _plot_results(data1, data2, colors, title, xlabel, ylabel):
    plt.figure(figsize=(6, 5), tight_layout=True)
    plt.scatter(data1[:, 0], data1[:, 1], c=colors[:len(data1)], alpha=0.2, label="Original")
    plt.scatter(data2[:, 0], data2[:, 1], c=colors[len(data1):], alpha=0.2, label="Synthetic")
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncol=2, frameon=False, fontsize=15)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(6))
    plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(6))
    plt.gca().xaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.2f}'))
    plt.gca().yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.2f}'))
    plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: '{:.2f}'.format(x)))
    plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: '{:.2f}'.format(y)))
    plt.show()