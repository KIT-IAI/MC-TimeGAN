"""mctimegan.py"""

# Import basic Python packages
import time
from itertools import chain, cycle

# Import additional packages
import numpy as np

# Import machine learning packages
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_worker():
    """Seeds the random number generator for reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

class ConditioningNetwork(nn.Module):
    """The Conditioning Network transforms input conditions using a 
    linear layer and a Tanh activation, followed by another linear layer.
    """

    def __init__(self, input_size, condition_size):
        super().__init__()
        self.condition = nn.Sequential(
            nn.Linear(input_size, 8),
            nn.Tanh(),
            nn.Linear(8, condition_size)
        )

    def forward(self, conds):
        """Forward pass for the conditioning network."""
        return self.condition(conds) if conds is not None else None

class Embedder(nn.Module):
    """The Embedder uses an RNN (GRU or LSTM) to process input sequences. 
    It consists of the RNN followed by a linear layer with a Sigmoid activation.
    """

    def __init__(self, module_name, input_features, hidden_dim, num_layers):
        super().__init__()
        assert module_name in ['gru', 'lstm']
        rnn_class = nn.GRU if module_name == 'gru' else nn.LSTM
        self.rnn = rnn_class(input_size=input_features, hidden_size=hidden_dim,
                             num_layers=num_layers, batch_first=True)
        self.model = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, x, c=None):
        """Forward pass for the embedding network."""
        if c is not None:
            x = torch.cat([x, c], dim=-1)  # Concatenate input with conditioning.
        seq, _ = self.rnn(x)  # Process sequence without conditioning.
        return self.model(seq)

class Recovery(nn.Module):
    """The Recovery network reconstructs the original input features from 
    the hidden states using an RNN and a linear layer with a Sigmoid activation.
    """

    def __init__(self, module_name, input_features, hidden_dim, num_layers):
        super().__init__()
        assert module_name in ['gru', 'lstm']
        rnn_class = nn.GRU if module_name == 'gru' else nn.LSTM
        self.rnn = rnn_class(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers,
                             batch_first=True)
        self.model = nn.Sequential(
            nn.Linear(hidden_dim, input_features),
            nn.Sigmoid()
        )

    def forward(self, x, c=None):
        """Forward pass for the recovery network."""
        if c is not None:
            x = torch.cat([x, c], dim=-1)  # Concatenate input with conditioning.
        seq, _ = self.rnn(x)  # Process sequence without conditioning.
        return self.model(seq)

class Generator(nn.Module):
    """The Generator generates synthetic sequences from random noise 
    using an RNN and a linear layer with a Sigmoid activation.
    """

    def __init__(self, module_name, input_features, hidden_dim, num_layers):
        super().__init__()
        assert module_name in ['gru', 'lstm']
        rnn_class = nn.GRU if module_name == 'gru' else nn.LSTM
        self.rnn = rnn_class(input_size=input_features, hidden_size=hidden_dim,
                             num_layers=num_layers,batch_first=True)
        self.model = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, x, c=None):
        """Forward pass for the generator network."""
        if c is not None:
            x = torch.cat([x, c], dim=-1)  # Concatenate input with conditioning.
        seq, _ = self.rnn(x)  # Process sequence without conditioning.
        return self.model(seq)

class Supervisor(nn.Module):
    """The Supervisor network is a simplified RNN (GRU or LSTM) that aids the generator 
    in producing more realistic sequences. It uses one fewer layer than the generator.
    """

    def __init__(self, module_name, input_features, hidden_dim, num_layers):
        super().__init__()
        assert module_name in ['gru', 'lstm']
        rnn_class = nn.GRU if module_name == 'gru' else nn.LSTM
        self.rnn = rnn_class(input_size=input_features, hidden_size=hidden_dim,
                             num_layers=num_layers-1,batch_first=True)
        self.model = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, x, c=None):
        """Forward pass for the supervisor network."""
        if c is not None:
            x = torch.cat([x, c], dim=-1)  # Concatenate input with conditioning.
        seq, _ = self.rnn(x)  # Process sequence without conditioning.
        return self.model(seq)

class Discriminator(nn.Module):
    """The Discriminator evaluates the authenticity of sequences using a bidirectional 
    RNN and a linear layer. The output is a single value indicating the realness of 
    the sequence.
    """

    def __init__(self, module_name, hidden_dim, num_layers):
        super().__init__()
        assert module_name in ['gru', 'lstm']
        rnn_class = nn.GRU if module_name == 'gru' else nn.LSTM
        self.rnn = rnn_class(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers,
                             bidirectional=True, batch_first=True)
        self.model = nn.Linear(2 * hidden_dim, 1)

    def forward(self, x, c=None):
        """Forward pass for the discriminator network."""
        if c is None:
            seq, _ = self.rnn(x)  # Process sequence without conditioning.
        else:
            x = torch.cat([x, c], dim=-1)  # Concatenate input with conditioning.
            seq, _ = self.rnn(x)
        return self.model(seq)

# Defines loss functions for the GAN components
# including discriminator loss, generator loss, and embedder loss.
def discriminator_loss(y_real, y_fake, y_fake_e):
    """Computes the loss for the discriminator."""
    gamma = 1
    valid = torch.ones_like(y_real, dtype=torch.float32, device=device, requires_grad=False)
    fake = torch.zeros_like(y_fake, dtype=torch.float32, device=device, requires_grad=False)

    d_loss_real = nn.BCEWithLogitsLoss()(y_real, valid)
    d_loss_fake = nn.BCEWithLogitsLoss()(y_fake, fake)
    d_loss_fake_e = nn.BCEWithLogitsLoss()(y_fake_e, fake)

    return d_loss_real + d_loss_fake + d_loss_fake_e * gamma

def generator_loss(y_fake, y_fake_e, h, h_hat_supervise, x, x_hat):
    """Computes the loss for the generator."""
    gamma = 1
    fake = torch.ones_like(y_fake, dtype=torch.float32, device=device, requires_grad=False)
    # 1. Unsupervised generator loss
    g_loss_u = nn.BCEWithLogitsLoss()(y_fake, fake)
    g_loss_u_e = nn.BCEWithLogitsLoss()(y_fake_e, fake)
    # 2. Supervised loss
    g_loss_s = nn.MSELoss()(h[:, 1:, :], h_hat_supervise[:, :-1, :])
    # 3. Two moments
    g_loss_v1 = torch.mean(torch.abs(torch.std(x_hat, dim=0) - torch.std(x, dim=0)))
    g_loss_v2 = torch.mean(torch.abs(torch.mean(x_hat, dim=0) - torch.mean(x, dim=0)))
    g_loss_v = g_loss_v1 + g_loss_v2

    return g_loss_u + gamma * g_loss_u_e + 100 * torch.sqrt(g_loss_s) + 100 * g_loss_v

def embedder_loss(x, x_tilde):
    """Computes the loss for the embedder."""
    return 10 * torch.sqrt(nn.MSELoss()(x, x_tilde))

def generator_loss_supervised(h, h_hat_supervise):
    """Computes the supervised loss for the generator."""
    return nn.MSELoss()(h[:, 1:, :], h_hat_supervise[:, :-1, :])

class MCTimeGAN(nn.Module):
    """
    MCTimeGAN: A Conditional Multivariate Time Series Generative Adversarial Network
    This class implements a conditional GAN for generating synthetic time series data.
    It includes the following components:
    - Embedding Network: Transforms the input time series data into a latent space.
    - Recovery Network: Recovers the original data from the latent space representation.
    - Generator Network: Generates synthetic data from random noise.
    - Supervisor Network: Supervises the generator during training to ensure temporal consistency.
    - Discriminator Network: Distinguishes between real and generated data.

    The training process consists of three phases:
    1. Embedding Network Training: Trains the embedding and recovery networks.
    2. Supervised Generator Training: Trains the generator with supervised loss.
    3. Joint Training: Alternates between training the generator and discriminator.

    The class also includes methods for transforming new data using the trained generator and for
    reporting training losses.
    Structure of our MC-TimeGAN implementation used in the synthetic grid congestion framework.
    The architecture includes Conditioning Network (C), Embedder (E), Recovery (R), Supervisor (S),
    Generator (G), and Discriminator (D).
    Dashed lines indicate the concatenation of the representation vectors of the labels wt with
    the input sequences for conditional network processing.
    """
    def __init__(self, module_name='gru', input_features=1, input_conditions=None, hidden_dim=8,
                 num_layers=3, epochs=100, batch_size=128, learning_rate=1e-3):
        super().__init__()
        self.module_name = module_name
        self.input_features = input_features
        self.input_conditions = input_conditions
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.cond_size = 1

        self.reproducibility = False
        # Initialize network components depending on whether input_conditions are provided
        if input_conditions is not None:
            self.condnet = ConditioningNetwork(input_conditions, self.cond_size)
            self.embedder = Embedder(module_name, input_features + self.cond_size,
                                     hidden_dim, num_layers)
            self.recovery = Recovery(module_name, input_features, hidden_dim + self.cond_size,
                                     num_layers)
            self.generator = Generator(module_name, input_features + self.cond_size,
                                       hidden_dim, num_layers)
            self.supervisor = Supervisor(module_name, hidden_dim + self.cond_size,
                                         hidden_dim, num_layers)
            self.discriminator = Discriminator(module_name, hidden_dim + self.cond_size,
                                               num_layers)
            # Define optimizers for different network components
            self.optimizer_e = torch.optim.Adam(
                chain(self.condnet.parameters(), self.embedder.parameters(),
                      self.recovery.parameters()),lr=learning_rate
            )
            self.optimizer_g = torch.optim.Adam(
                chain(self.condnet.parameters(), self.generator.parameters(),
                      self.supervisor.parameters()), lr=learning_rate
            )
            self.optimizer_d = torch.optim.Adam(
                chain(self.condnet.parameters(), self.discriminator.parameters()), lr=learning_rate
            )
        else:
            self.embedder = Embedder(module_name, input_features, hidden_dim, num_layers)
            self.recovery = Recovery(module_name, input_features, hidden_dim, num_layers)
            self.generator = Generator(module_name, input_features, hidden_dim, num_layers)
            self.supervisor = Supervisor(module_name, hidden_dim, hidden_dim, num_layers)
            self.discriminator = Discriminator(module_name, hidden_dim, num_layers)
            # Define optimizers for different network components
            self.optimizer_e = torch.optim.Adam(
                chain(self.embedder.parameters(), self.recovery.parameters()), lr=learning_rate
            )
            self.optimizer_g = torch.optim.Adam(
                chain(self.generator.parameters(), self.supervisor.parameters()), lr=learning_rate
            )
            self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate)

        self.fitting_time = None
        self.losses = []

    def fit(self, data_train: np.ndarray, **kwargs: np.ndarray):
        """Fits the MC-TimeGAN model to the training data."""
        self.fitting_time = time.time()
        # Convert training data to tensor and move to device
        data_train = torch.tensor(data_train, dtype=torch.float32, device=device)

        conditions = np.concatenate([c for c in kwargs.values()], axis=-1) if kwargs else None
        conditions = torch.tensor(conditions, dtype=torch.float32,
                                  device=device) if kwargs else None

        dataset = TensorDataset(data_train, conditions) if kwargs else TensorDataset(data_train)

        # 1. Embedding network training
        print('Start Embedding Network Training')
        for epoch, frame in zip(range(self.epochs), cycle(r'-\|/-\|/')):
            # Create DataLoader for training
            batches_train = DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True,
                worker_init_fn=seed_worker if self.reproducibility else None,
                generator=g if self.reproducibility else None
            )

            self.train()
            loss_e = []
            for batch in batches_train:
                x, c = batch if kwargs else (*batch, None)

                self.optimizer_e.zero_grad()
                # Generate conditioning variables
                conds = self.condnet(c) if c is not None else None
                h = self.embedder(x, conds)  # Embed the input data
                x_tilde = self.recovery(h, conds)  # Recover the input from the embedded data
                e_loss = embedder_loss(x, x_tilde)  # Calculate embedding loss

                e_loss.backward()
                self.optimizer_e.step()

                loss_e.append(e_loss.item())

            if (epoch + 1) % (0.1 * self.epochs) == 0:
                print(f'\rEpoch {epoch + 1} of {self.epochs} | loss_e {np.mean(loss_e):12.9f}')
            else:
                print(f'\r{frame}', sep='', end='', flush=True)

        print('Finished Embedding Network Training')

        #2. Training using only supervised loss
        print('Start Training with Supervised Loss Only')
        for epoch, frame in zip(range(self.epochs), cycle(r'-\|/-\|/')):
            # Create DataLoader for training
            batches_train = DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True,
                worker_init_fn=seed_worker if self.reproducibility else None,
                generator=g if self.reproducibility else None
            )

            self.train()
            loss_g = []
            for batch in batches_train:
                x, c = batch if kwargs else (*batch, None)

                self.optimizer_g.zero_grad()
                # Generate conditioning variables
                conds = self.condnet(c) if c is not None else None

                h = self.embedder(x, conds)  # Embed the input data
                h_hat_supervise = self.supervisor(h, conds)  # Supervise the embedded data
                g_loss = generator_loss_supervised(h, h_hat_supervise)  # Calculate generator loss

                g_loss.backward()
                self.optimizer_g.step()

                loss_g.append(g_loss.item())

            if (epoch + 1) % (0.1 * self.epochs) == 0:
                print(f'\rEpoch {epoch + 1} of {self.epochs} | loss_g {np.mean(loss_g):12.9f}')
            else:
                print(f'\r{frame}', sep='', end='', flush=True)

        print('Finished Training with Supervised Loss Only')

        # Joint training
        print('Start Joint Training')
        for epoch, frame in zip(range(self.epochs), cycle(r'-\|/-\|/')):
            #Traing generator twice more than discriminator
            loss_g = []
            loss_e = []
            # Perhaps training can be further improved by training the generator even more often
            # default = 2
            for _ in range(2):
                data_z = torch.rand(data_train.shape, dtype=torch.float32, device=device)

                dataset = TensorDataset(data_train, data_z,
                                         conditions) if kwargs else TensorDataset(data_train,
                                                                                  data_z)

                batches_train = DataLoader(
                    dataset, batch_size=self.batch_size, shuffle=True,
                    worker_init_fn=seed_worker if self.reproducibility else None,
                    generator=g if self.reproducibility else None
                )

                self.train()
                for batch in batches_train:
                    x, z, c = batch if kwargs else (*batch, None)
                    self.optimizer_g.zero_grad()
                    # Generate conditioning variables
                    conds = self.condnet(c) if c is not None else None
                    # Embed the input data
                    h = self.embedder(x, conds)
                    # Generate synthetic data
                    e_hat = self.generator(z, conds)
                    # Supervise the generated data
                    h_hat = self.supervisor(e_hat, conds)
                    # Supervise the embedded data
                    h_hat_supervise = self.supervisor(h, conds)
                    # Recover the input from the supervised data
                    x_hat = self.recovery(h_hat, conds)
                    # Discriminator output for supervised data
                    y_fake = self.discriminator(h_hat, conds)
                    # Discriminator output for generated data
                    y_fake_e = self.discriminator(e_hat, conds)
                    # Calculate generator loss
                    g_loss = generator_loss(y_fake, y_fake_e, h, h_hat_supervise, x, x_hat)

                    g_loss.backward()
                    self.optimizer_g.step()

                    loss_g.append(g_loss.item())

                    self.optimizer_e.zero_grad()
                    # Generate conditioning variables
                    conds = self.condnet(c) if c is not None else None

                    h = self.embedder(x, conds)  # Embed the input data
                    h_hat_supervise = self.supervisor(h, conds)  # Supervise the embedded data
                    x_tilde = self.recovery(h, conds)  # Recover the input from the embedded data
                    # Calculate embedding loss
                    embed_loss = embedder_loss(x, x_tilde)
                    gen_loss_sup = 0.1 * generator_loss_supervised(h, h_hat_supervise)
                    e_loss = embed_loss + gen_loss_sup
                    e_loss.backward()
                    self.optimizer_e.step()

                    loss_e.append(e_loss.item())

            data_z = torch.rand(data_train.shape, dtype=torch.float32, device=device)

            dataset = TensorDataset(data_train, data_z,
                                    conditions) if kwargs else TensorDataset(data_train,
                                                                             data_z)

            batches_train = DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True,
                worker_init_fn=seed_worker if self.reproducibility else None,
                generator=g if self.reproducibility else None
            )

            self.train()
            loss_d = []
            for batch in batches_train:
                x, z, c = batch if kwargs else (*batch, None)
                self.optimizer_d.zero_grad()
                # Generate conditioning variables
                conds = self.condnet(c) if c is not None else None
                # Embed the input data
                h = self.embedder(x, conds)
                # Generate synthetic data
                e_hat = self.generator(z, conds)
                # Supervise the generated data
                h_hat = self.supervisor(e_hat, conds)
                # Discriminator output for supervised data
                y_fake = self.discriminator(h_hat, conds)
                # Discriminator output for real data
                y_real = self.discriminator(h, conds)
                # Discriminator output for generated data
                y_fake_e = self.discriminator(e_hat, conds)
                # Calculate discriminator loss
                d_loss = discriminator_loss(y_real, y_fake, y_fake_e)
                loss_d.append(d_loss.item())
                if d_loss > 0.15:
                    d_loss.backward()
                    self.optimizer_d.step()
            # Store losses for each epoch
            self.losses.append([epoch + 1, np.mean(loss_g),
                                np.mean(loss_e),
                                 np.mean(loss_d)])
            if (epoch + 1) % (0.1 * self.epochs) == 0:
                print(
                    f'\rEpoch {epoch + 1} of {self.epochs} | loss_g {np.mean(loss_g):12.9f} | '
                    f'loss_e {np.mean(loss_e):12.9f} | loss_d {np.mean(loss_d):12.9f}'
                )
            else:
                print(f'\r{frame}', sep='', end='', flush=True)
        # Mark the model as fitted
        self.fitting_time = np.round(time.time() - self.fitting_time, 3)

    def transform(self, data_shape, **kwargs):
        """Generates synthetic data using the trained model."""
        # Generate random noise for the generator input
        data_z = torch.rand(size=data_shape, dtype=torch.float32,
                            device=device, requires_grad=False)

        conditions = np.concatenate([c for c in kwargs.values()],
                                     axis=-1) if kwargs else None
        conditions = torch.tensor(conditions, dtype=torch.float32,
                                  device=device, requires_grad=False) if kwargs else None

        dataset = TensorDataset(data_z, conditions) if kwargs else TensorDataset(data_z)

        batches = DataLoader(
            dataset, batch_size=1,
            worker_init_fn=seed_worker if self.reproducibility else None,
            generator=g if self.reproducibility else None
        )

        generated_data = []
        self.eval()
        with torch.no_grad():
            for batch in batches:
                z, c = batch if kwargs else (*batch, None)
                # Generate conditioning variables
                conds = self.condnet(c) if c is not None else None

                e_hat = self.generator(z, conds)  # Generate synthetic data
                h_hat = self.supervisor(e_hat, conds)  # Supervise the generated data
                x_hat = self.recovery(h_hat, conds)  # Recover the input from the supervised data

                generated_data.append(np.squeeze(x_hat.cpu().numpy(), axis=0))

        return np.stack(generated_data)
