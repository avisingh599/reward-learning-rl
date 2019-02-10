"""Autoencoders for images of size (84, 84, 3)."""

import numpy as np
import torch
from torch.autograd import Variable
from torch.distributions import Normal
from torch import nn
import torch.nn.functional as nnf


class VAE(nn.Module):
    """Variational autoencoder for images of size (84,  84, 3)."""

    def __init__(self, num_dims=4, min_variance=1e-4):
        self.min_variance = min_variance
        self.num_dims = num_dims
        self.log_min_variance = float(np.log(min_variance))
        nn.Module.__init__(self)

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=5, stride=3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=5, stride=3),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )
        self.mu_fc = nn.Linear(128, self.num_dims)
        self.logvar_fc = nn.Linear(128, self.num_dims)
        self.expand_fc = nn.Sequential(
            nn.Linear(self.num_dims, 128),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, kernel_size=6, stride=3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 3, kernel_size=6, stride=3),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), 128)
        mu = self.mu_fc(x)
        logvar = self.logvar_fc(x)
        if self.min_variance is not None:
            logvar = self.log_min_variance + torch.abs(logvar)
        return mu, logvar

    def decode(self, x):
        x = self.expand_fc(x)
        x = x.view(x.size(0), 32, 2, 2)
        return self.decoder(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        z = Variable(torch.randn(std.shape).cuda())
        return mu + std * z

    def features(self, x):
        mu, _logvar = self.encode(x)
        return mu
        # return torch.cat((mu, logvar), dim=1)

    def forward(self, x):
        mu, logvar = self.encode(x)
        x = self.reparameterize(mu, logvar)
        return self.decode(x), mu, logvar

    def reconstruction(self, x):
        reconstruction, _mu, _logvar = self.forward(x)
        return reconstruction

    def loss(self, x):
        reconstruction, mu, logvar = self.forward(x)
        BCE = nnf.binary_cross_entropy(reconstruction, x, size_average=False)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar))
        return BCE + KLD


class AE(nn.Module):
    """Vanilla autoencoder for images of size (64, 64, 3)."""

    def __init__(self):
        nn.Module.__init__(self)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=5, stride=3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=5, stride=3),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )
        self.features_fc = nn.Linear(128, 32)
        self.expand_fc = nn.Linear(32, 128)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=3),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=6, stride=3),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=6, stride=3),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), 128)
        x = self.features_fc(x)
        return x

    def decode(self, x):
        x = self.expand_fc(x)
        x = x.view(x.size(0), 32, 2, 2)
        return self.decoder(x)

    def features(self, x):
        return self.encode(x)

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def reconstruction(self, x):
        return self.forward(x)

    def loss(self, x):
        #criterion = nn.MSELoss()
        BCE = nnf.binary_cross_entropy(self.forward(x), x, size_average=False)
        return BCE
#return criterion(self.forward(x), x)