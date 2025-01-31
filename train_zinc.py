import argparse
import os
import numpy as np

from modules.model_zinc import MoleculeVAE
from modules.callbacks import ModelCheckpoint, ReduceLROnPlateau

import h5py
from zinc_grammar import gram

rules = gram.split('\n')
MAX_LEN = 277
DIM = len(rules)
BATCH = 500


def train_znic(args):
    np.random.seed(1)
    # 1. get any arguments and define save file, then create the VAE model

    print('L=' + str(args.latent_dim) + ' E=' + str(args.epochs))
    model_save = 'results/zinc_vae_grammar_L' + str(args.latent_dim) + '_E' + str(args.epochs) + '_val.hdf5'
    print(model_save)
    model = MoleculeVAE()
    print(args.load_model)

    # 2. if this results file exists already load it
    if os.path.isfile(args.load_model):
        print('loading!')
        model.load(rules, args.load_model, latent_rep_size=args.latent_dim, max_length=MAX_LEN)
    else:
        print('making new model')
        model.create(rules, max_length=MAX_LEN, latent_rep_size=args.latent_dim)

    # 3. load dataset
    h5f = h5py.File('data/zinc_grammar_dataset.h5', 'r')
    data = h5f['data'][:]
    h5f.close()

    # 3.1. split into train/test, we use test set to check reconstruction error and the % of
    # samples from prior p(z) that are valid
    train_data = data[0:5000]
    # valid_data = data[5000:]

    # 4. only save best model found on a 10% validation set
    checkpointer = ModelCheckpoint(filepath=model_save, verbose=1, save_best_only=True)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)
    # 5. fit the vae
    model.AE.fit(train_data,  train_data, shuffle=True, nb_epoch=args.epochs,
                 batch_size=BATCH, callbacks=[checkpointer, reduce_lr],
                 validation_split=0.1)


if __name__ == '__main__':
    LATENT = 56
    EPOCHS = 100
    parser = argparse.ArgumentParser(description='Molecular AE')
    parser.add_argument('--load_model', type=str, metavar='N', default="")
    parser.add_argument('--epochs', type=int, metavar='N', default=EPOCHS,
                        help='Number of epochs to for train.')
    parser.add_argument('--latent_dim', type=int, metavar='N', default=LATENT, help='Dim of latent represent.')
    args = parser.parse_args()
    train_znic(args)
