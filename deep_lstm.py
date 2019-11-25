import time
import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell

from utils import array_pad
from base_model import Model
from cells import LSTMCell, MultiRNNCellWithSkipConn
from data_utils import load_vocab, load_dataset

import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class DeepLSTM(nn.Module):
  """Deep LSTM model."""
  def __init__(self, size=256, depth=3, batch_size=32,
               keep_prob=0.1, max_nsteps=1000,
               checkpoint_dir="checkpoint", forward_only=False):
    """Initialize the parameters for an Deep LSTM model.
    
    Args:
      size: int, The dimensionality of the inputs into the Deep LSTM cell [32, 64, 256]
      learning_rate: float, [1e-3, 5e-4, 1e-4, 5e-5]
      batch_size: int, The size of a batch [16, 32]
      keep_prob: unit Tensor or float between 0 and 1 [0.0, 0.1, 0.2]
      max_nsteps: int, The max time unit [1000]
    """
    super(DeepLSTM, self).__init__()

    self.size = int(size)
    self.depth = int(depth)
    self.batch_size = int(batch_size)
    self.output_size = self.depth * self.size
    self.keep_prob = float(keep_prob)
    self.max_nsteps = int(max_nsteps)
    self.checkpoint_dir = checkpoint_dir

    start = time.clock()
    print(" [*] Building Deep LSTM...")
    self.cell = LSTMCell(size, forget_bias=0.0)

    if not forward_only and self.keep_prob < 1:
      # self.cell = rnn_cell.DropoutWrapper(self.cell, output_keep_prob=keep_prob)
      ### TODO
      d = nn.Dropout(p=keep_prob)
      self.cell = d(self.cell)
    self.stacked_cell = MultiRNNCellWithSkipConn([self.cell] * depth)

    self.initial_state = self.stacked_cell.zero_state(batch_size, torch.float32)

  def prepare_model(self, data_dir, dataset_name, vocab_size):
    if not self.vocab:
      self.vocab, self.rev_vocab = load_vocab(data_dir, dataset_name, vocab_size)
      print(" [*] Loading vocab finished.")

    self.vocab_size = len(self.vocab)

    self.emb = nn.Embedding(self.vocab_size, self.size)

    # inputs
    self.inputs = torch.IntTensor(self.batch_size, self.max_nsteps).zero_()
    embed_inputs = self.emb(torch.transpose(self.inputs, 1, 0))

    #tf.histogram_summary("embed", self.emb)
    ### TODO

    # output states
    # _, states = rnn.rnn(self.stacked_cell,
    #                     tf.unpack(embed_inputs),
    #                     dtype=tf.float32,
    #                     initial_state=self.initial_state)
    _, states = rnn.rnn(self.stacked_cell,
                        torch.unbind(embed_inputs),
                        dtype=torch.float32,
                        initial_state=self.initial_state)
    self.batch_states = torch.stack(states)

    self.nstarts = torch.IntTensor(self.batch_size, 3).zero_()
    # outputs = tf.pack([tf.slice(self.batch_states, nstarts, [1, 1, self.output_size])
    #     for idx, nstarts in enumerate(tf.unpack(self.nstarts))])
    ### TODO
    outputs = torch.stack(self.batch_states[:2, :2, :self.output_size+1] for idx, nstarts in enumerate(torch.unbind(self.nstarts)))

    self.outputs = outputs.view(self.batch_size, self.output_size).size()

    self.W = torch.randn((self.vocab_size, self.output_size), requires_grad=True)
    # tf.histogram_summary("weights", self.W)
    # tf.histogram_summary("output", outputs)
    ### TODO

    # logits
    self.y = torch.FloatTensor(self.batch_size, self.vocab_size).zero_()
    # labels
    self.y_ = torch.matmul(self.outputs, torch.transpose(self.W, 1, 0))

    loss_fn = torch.nn.CrossEntropyLoss()
    self.loss = loss_fn(self.y_, self.y)
    #tf.scalar_summary("loss", tf.reduce_mean(self.loss))
    ### TODO

    _, logits_indices = torch.max(self.y, 1)
    _, labels_indices = torch.max(self.y_, 1)
    x = list(logits_indices.size())[0]
    y = list(labels_indices.size())[0]
    correct_prediction = logits_indices.expand(x,y).eq(labels_indices.expand(x,y))
    self.accuracy = torch.mean(correct_prediction.type(torch.FloatTensor))
    
    #tf.scalar_summary("accuracy", self.accuracy)
    ### TODO

    print(" [*] Preparing model finished.")
