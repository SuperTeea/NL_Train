import collections
from torch import nn
import torch
import torch.nn.functional as F
import inspect

class Seq2Seq(nn.Module):  #@save
    """The base class for the encoder--decoder architecture."""
    def __init__(self, encoder, decoder, tgt_pad, lr):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.tgt_pad = tgt_pad
        self.lr = lr

    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)

    def configure_optimizers(self):
        # Adam optimizer is used here
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def loss(self, Y_hat, Y):
        l = super().loss(Y_hat, Y, averaged=False)
        mask = (Y.reshape(-1) != self.tgt_pad).type(torch.float32)
        return (l * mask).sum() / mask.sum()
    
class HyperParameters:
    """The base class of hyperparameters."""
    def save_hyperparameters(self, ignore=[]):
        """Defined in :numref:`sec_oo-design`"""
        raise NotImplemented

    def save_hyperparameters(self, ignore=[]):
        """Save function arguments into class attributes.
    
        Defined in :numref:`sec_utils`"""
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {k:v for k, v in local_vars.items()
                        if k not in set(ignore+['self']) and not k.startswith('_')}
        for k, v in self.hparams.items():
            setattr(self, k, v)
    
class Classifier(nn.Module):
    """The base class of classification models.

    Defined in :numref:`sec_classification`"""
    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        # self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)
        # self.plot('acc', self.accuracy(Y_hat, batch[-1]), train=False)
        print('loss', self.loss(Y_hat, batch[-1]))
        print('acc', self.accuracy(Y_hat, batch[-1]))

    def accuracy(self, Y_hat, Y, averaged=True):
        """Compute the number of correct predictions.
    
        Defined in :numref:`sec_classification`"""
        Y_hat = torch.reshape(Y_hat, (-1, Y_hat.shape[-1]))
        preds = torch.astype(torch.argmax(Y_hat, axis=1), Y.dtype)
        compare = torch.astype(preds == torch.reshape(Y, -1), torch.float32)
        return torch.reduce_mean(compare) if averaged else compare

    def loss(self, Y_hat, Y, averaged=True):
        """Defined in :numref:`sec_softmax_concise`"""
        Y_hat = torch.reshape(Y_hat, (-1, Y_hat.shape[-1]))
        Y = torch.reshape(Y, (-1,))
        return F.cross_entropy(
            Y_hat, Y, reduction='mean' if averaged else 'none')

    def layer_summary(self, X_shape):
        """Defined in :numref:`sec_lenet`"""
        X = torch.randn(*X_shape)
        for layer in self.net:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape:\t', X.shape)
            
class Module(nn.Module, HyperParameters):
    """The base class of models.

    Defined in :numref:`sec_oo-design`"""
    def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
        super().__init__()
        self.save_hyperparameters()
        # self.board = ProgressBoard()

    def loss(self, y_hat, y):
        raise NotImplementedError

    def forward(self, X):
        assert hasattr(self, 'net'), 'Neural network is defined'
        return self.net(X)

    # def plot(self, key, value, train):
    #     """Plot a point in animation."""
    #     assert hasattr(self, 'trainer'), 'Trainer is not inited'
    #     self.board.xlabel = 'epoch'
    #     if train:
    #         x = self.trainer.train_batch_idx / \
    #             self.trainer.num_train_batches
    #         n = self.trainer.num_train_batches / \
    #             self.plot_train_per_epoch
    #     else:
    #         x = self.trainer.epoch + 1
    #         n = self.trainer.num_val_batches / \
    #             self.plot_valid_per_epoch
    #     self.board.draw(x, torch.numpy(torch.to(value, torch.cpu())),
    #                     ('train_' if train else 'val_') + key,
    #                     every_n=int(n))

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=False)

    def configure_optimizers(self):
        raise NotImplementedError

    def configure_optimizers(self):
        """Defined in :numref:`sec_classification`"""
        return torch.optim.SGD(self.parameters(), lr=self.lr)

    def apply_init(self, inputs, init=None):
        """Defined in :numref:`sec_lazy_init`"""
        self.forward(*inputs)
        if init is not None:
            self.net.apply(init)