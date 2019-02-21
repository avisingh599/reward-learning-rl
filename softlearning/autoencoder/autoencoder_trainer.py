"""Train and save an autoencoder."""

import argparse
import os
import os.path as osp
import pickle
import torch
import numpy as np
from tqdm import trange
from softlearning.autoencoder.autoencoder import AE, VAE
from scipy.misc import imsave
try:
    from torch.nn.functional import Variable
except:
    from torch.autograd.variable import Variable


class AutoencoderTrainer(object):
    """Class for training autoencoders."""

    def __init__(
            self,
            model,
            train_dataset,
            test_dataset,
            test_dir,
            save_path,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.test_dir = test_dir
        self.save_path = save_path

    @staticmethod
    def split_dataset(dataset, train_fraction=0.9):
        n_train = int(len(dataset) * train_fraction)
        return dataset[:n_train], dataset[n_train:]

    def _get_batch(self, batch_size, training=True):
        dataset = self.train_dataset if training else self.test_dataset
        indices = np.random.randint(0, len(dataset), batch_size)
        samples = dataset[indices, :].transpose((0, 3, 1, 2))
        return Variable(torch.from_numpy(samples.astype(np.float32)).cuda())

    def _train_epoch(self, epoch, optimizer, batch_size, n_val=5, val_every=50):
        train_loss = 0.0
        progressbar = trange(len(self.train_dataset) // batch_size)

        for i in progressbar:
            batch_images = self._get_batch(batch_size, training=True)
            loss = self.model.loss(batch_images)

            optimizer.zero_grad()
            loss.backward()
            loss_np = loss.cpu().data.numpy()
            # train_loss += loss.cpu().data.numpy()[0]
            if isinstance(loss_np, np.ndarray) and loss_np.shape != ():
                train_loss += loss_np[0]
            else:
                train_loss += float(loss_np)
            optimizer.step()

            avg_train_loss = train_loss / (i+1) / batch_size
            progressbar.set_description('[{}] loss: {:.8f}'.format(
                epoch, avg_train_loss
            ))

        if epoch is not None and (epoch + 1) % val_every == 0:
            self.test(epoch=epoch, n=n_val)
            self.save(self.save_path)

        print('epoch: {}'.format(epoch))
        print('loss: {}'.format(avg_train_loss))
        #logger.record_tabular('epoch', epoch)
        #logger.record_tabular('loss', avg_train_loss)

    def train(self, num_epochs=20, learning_rate=0.001, batch_size=64, n_val=5, val_every=50):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            self._train_epoch(epoch, optimizer, batch_size, n_val=n_val, val_every=val_every)

    def save(self, save_path):
        torch.save(self.model.state_dict(), save_path)

    def test(self, epoch=None, n=1):
        if epoch is None:
            epoch_dir = self.test_dir
        else:
            epoch_dir = osp.join(self.test_dir, 'epoch{}'.format(epoch))
            if not osp.isdir(epoch_dir):
                os.makedirs(epoch_dir)

        batch_images = self._get_batch(n, training=False)
        reconstruction = self.model.reconstruction(batch_images)

        batch_images = batch_images.cpu().data.numpy()
        reconstruction = reconstruction.cpu().data.numpy()

        batch_images = batch_images.transpose(0, 2, 3, 1)
        reconstruction = reconstruction.transpose(0, 2, 3, 1)

        for i in range(n):
            image_path = osp.join(epoch_dir, 'image{}.png'.format(i))
            reconstruction_path = osp.join(
                epoch_dir,
                'reconstruction{}.png'.format(i)
            )
            imsave(image_path, batch_images[i])
            imsave(reconstruction_path, reconstruction[i])
