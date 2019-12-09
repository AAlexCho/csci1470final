import torch
import torch.nn as nn
from torch.nn.RNNCell import RNNCell, linear

import torch.nn.functional as F
import os
import numpy as np
class MultiRNNCellWithSkipConn(RNNCell):
  """Almost same with tf.models.rnn.rnn_cell.MultiRnnCell adding
  a skip connection from the input of current time t and using
  _num_units not state size because LSTMCell returns only [h] not [c, h].
  """

  def __init__(self, cells):
    """Create a RNN cell composed sequentially of a number of RNNCells.
    Args:
      cells: list of RNNCells that will be composed in this order.
    Raises:
      ValueError: if cells is empty (not allowed) or if their sizes don't match.
    """
    if not cells:
      raise ValueError("Must specify at least one cell for MultiRNNCell.")
    for i in range(len(cells) - 1):
      if cells[i + 1].input_size != cells[i].output_size:
        raise ValueError("In MultiRNNCell, the input size of each next"
                         " cell must match the output size of the previous one."
                         " Mismatched output size in cell %d." % i)
    self._cells = cells

  @property
  def input_size(self):
    return self._cells[0].input_size

  @property
  def output_size(self):
    return self._cells[-1].output_size

  @property
  def state_size(self):
    return sum([cell.state_size for cell in self._cells])

  def __call__(self, inputs, state, scope=None):
    """Run this multi-layer cell on inputs, starting from state."""
	cur_state_pos = 0
	first_layer_input = cur_inp = inputs
	new_states = []
	for i, cell in enumerate(self._cells):
		#with tf.variable_scope("Cell%d" % i):
		cur_state = state[0: cur_state_pos, -1: cell.state_size]
		cur_state_pos += cell.state_size
		# Add skip connection from the input of current time t.
		if i != 0:
			first_layer_input = first_layer_input
		else:
			first_layer_input = torch.zeros_like(first_layer_input)
		cur_inp, new_state = cell(torch.cat(inputs), cur_state)
		new_states.append(new_state)
	return cur_inp, torch.cat(new_states)
