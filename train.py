
import time
import sys
import numpy as np
import torch as torch
from torch.utils.tensorboard import SummaryWriter
from utils import array_pad
import DeepLSTM

#from base_model import Model
#from cells import LSTMCell, MultiRNNCellWithSkipConn

#Changing imports to pytorch

import torch.nn as nn
import torch.optim as optim

def train(self, model, vocab_size, epoch=25, data_dir="data", dataset_name="cnn"):
    self.prepare_model(data_dir, dataset_name, vocab_size)

    y = np.zeros([self.batch_size, self.vocab_size])

    start_time = time.time()
    loss = nn.CrossEntropyLoss()

    for epoch_idx in range(epoch):

        batch_stop = False
        while True:
            y.fill(0)
            inputs, num_starts, answers = [], [], []
            batch_idx = 0
            while True:
                try:
                    (_, document, question, answer, _), data_idx, data_max_idx = data_loader.next()
                except StopIteration:
                    batch_stop = True
                    break

            # [0] means splitter between d and q
                data = [int(d) for d in document.split()] + [0] + \
                    [int(q) for q in question.split() for q in question.split()]

                if len(data) > self.max_nsteps:
                    continue

                inputs.append(data)
                num_starts.append(len(inputs[-1]) - 1)
                y[batch_idx][int(answer)] = 1

                batch_idx += 1
                if batch_idx == self.batch_size:
                    break
            if batch_stop:
                break

            FORCE = False
            if FORCE:
                inputs = array_pad(inputs, self.max_nsteps, pad=-1, force=FORCE)
                num_starts = np.where(inputs == -1)[1]
                inputs[inputs == -1] = 0
            else:
                inputs = array_pad(inputs, self.max_nsteps, pad=0)
            num_starts = [[num_starts, idx, 0] for idx, num_starts in enumerate(num_starts)]

            logits = model.prepareModel(data_dir, dataset_name, vocab_size)  # not probabilities!
            los = loss(logits, torch.tensor(np.array(inputs)))
            model.optimizer.zero_grad()
            los.backward()
            model.optimizer.step()  # can be thought of as gradient descent


