import torch
import torch.nn as nn
from torch.nn.RNNCell import RNNCell, linear
from torch.autograd import Variable

class LSTMCell(RNNCell):
  """Almost same with tf.models.rnn.rnn_cell.BasicLSTMCell
  except adding c to inputs and h to calculating gates,
  adding a skip connection from the input of current time t,
  and returning only h not concat of c and h.
  """

  def __init__(self, num_units, forget_bias=1.0):
    self._num_units = num_units
    self._forget_bias = forget_bias
    self.c = None

  @property
  def input_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Long short-term memory cell (LSTM)."""
    h = state
    if self.c == None:
      t = torch.zeros_like(self.c)
      self.c = t.view(-1, self._num_units)
    concat = linear([inputs, h, self.c], 4 * self._num_units, True)

    i, j, f, o = torch.split(concat, 4, 1)

    self.c = self.c * torch.sigmoid(f + self._forget_bias) + torch.sigmoid(i) * torch.tanh(j)
    new_h = torch.tanh(self.c) * torch.sigmoid(o)

   
    softmax_w = Variable(torch.randn(self._num_units, self._num_units).type(torch.FloatTensor), requires_grad=True)
    softmax_b = Variable(torch.randn(self._num_units).type(torch.FloatTensor), requires_grad=True)

    new_y = torch.add(torch.matmul(new_h, softmax_w), softmax_b)

    return new_y, new_y
