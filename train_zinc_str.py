from __future__ import print_function

import argparse
import os
import numpy as np

from models.model_zinc_str import MoleculeVAE
from models.callbacks import ModelCheckpoint, ReduceLROnPlateau

import h5py

charset = ['C', '(', ')', 'c', '1', '2', 'o', '=', 'O', 'N', '3', 'F', '[',
           '@', 'H', ']', 'n', '-', '#', 'S', 'l', '+', 's', 'B', 'r', '/',
           '4', '\\', '5', '6', '7', 'I', 'P', '8', ' ']

MAX_LEN = 120
DIM = len(charset)
BATCH = 500


def train_zinc_str(args):
    np.random.seed(1)
    # 1. get any arguments and define save file, then create the VAE model

    model_save = 'results/zinc_vae_str_L' + str(args.latent_dim) + '_E' + str(args.epochs) + '_val.hdf5'
    print(model_save)
    model = MoleculeVAE()
    print(args.load_model)

    # 2. if this results file exists already load it
    if os.path.isfile(args.load_model):
        print('loading model')
        model.load(charset, args.load_model, latent_rep_size=args.latent_dim)
    else:
        print('making new model')
        model.create(charset, max_length=MAX_LEN, latent_rep_size=args.latent_dim)

    print()
    # 3. only save best model found on a 10% validation set
    checkpointer = ModelCheckpoint(filepath=model_save, verbose=1, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)

    # 4. load dataset
    h5f = h5py.File('data/zinc_str_dataset.h5', 'r')
    data = h5f['data'][:]
    # 4.1. split into train/test, we use test set to check reconstruction error and the % of
    # samples from prior p(z) that are valid
    XTE = data[0:5000]
    XTR = data[5000:]
    h5f.close()

    # 5. fit the vae
    model.autoencoder.fit(XTE, XTE, shuffle=True,
                          nb_epoch=args.epochs, batch_size=BATCH,
                          callbacks=[checkpointer, reduce_lr],
                          validation_split=0.1)


if __name__ == '__main__':
    LATENT = 56
    EPOCHS = 100
    parser = argparse.ArgumentParser(description='Molecular AE network')
    parser.add_argument('--load_model', type=str, metavar='N', default="")
    parser.add_argument('--epochs', type=int, metavar='N', default=EPOCHS, help='Number of epochs for train.')
    parser.add_argument('--latent_dim', type=int, metavar='N', default=LATENT, help='Dim of latent represent.')
    args = parser.parse_args()
    train_zinc_str(args)
