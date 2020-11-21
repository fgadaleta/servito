from typing import List
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pickle
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from pprint import pprint
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
import tqdm
import sys
from config import BATCH_SIZE, \
    EMBEDDING_DIMS, \
    N_WORKERS, \
    N_EPOCHS, \
    AUTOENCODER_FILENAME, \
    KMEANS_FILENAME

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class Autoencoder(nn.Module):
    def __init__(self, input_dims, hidden_dims=[6, 9]):
        super(Autoencoder, self).__init__()
        self.hidden_dims = hidden_dims
        self.input_dims = input_dims

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dims, self.hidden_dims[0]),
            nn.ReLU(True),
            nn.Linear(self.hidden_dims[0], self.hidden_dims[1]),
            nn.ReLU(True),
            nn.Linear(self.hidden_dims[1], EMBEDDING_DIMS),
            )

        self.decoder = nn.Sequential(
            nn.Linear(EMBEDDING_DIMS, self.hidden_dims[1]),
            nn.ReLU(True),
            nn.Linear(self.hidden_dims[1], self.hidden_dims[0]),
            nn.ReLU(True),
            nn.Linear(self.hidden_dims[0], self.input_dims), nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, data):
        return self.encoder(torch.Tensor(data).double()).detach().numpy()


def train_autoencoder(data, input_dims, hidden_dims: List):
    # Train/Load autoencoder
    model = Autoencoder(input_dims, hidden_dims)
    model = model.double()
    criterion = nn.MSELoss()
    # criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

    if Path(AUTOENCODER_FILENAME).exists():
        print("Loading autoencoder from\t%s" % AUTOENCODER_FILENAME)
        model.load_state_dict(torch.load(AUTOENCODER_FILENAME))
    else:
        # Load and transform training data
        train_loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, num_workers=N_WORKERS)
        # TODO create test set
        # test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

        for epoch in range(N_EPOCHS):
            train_loss = 0.0
            tk0 = tqdm.tqdm(train_loader, total=int(len(train_loader)))
            counter = 0
            for batch, data_batch in enumerate(tk0):
                data_batch = data_batch.view(data_batch.size(0), -1)
                data_batch = Variable(data_batch)
                # ===================forward=====================
                output = model(data_batch.double())
                loss = criterion(output, data_batch)
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                # update running training loss
                train_loss += loss.item() * data_batch.size(0)
                optimizer.step()
                counter += 1
                tk0.set_postfix(loss=(train_loss / (counter * train_loader.batch_size)))

            # ===================log========================
            train_loss = train_loss/len(train_loader)
            print('epoch [{}/{}], loss:{:.4f}, train loss: {:.4f}'
                .format(epoch + 1, N_EPOCHS, loss.data.item(), train_loss))
        # Save autoencoder
        torch.save(model.state_dict(), AUTOENCODER_FILENAME)
    return model