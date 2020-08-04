import numpy as np
import tensorflow as tf

from tensorflow.keras import Model

EPS = 1e-8

LOG_STD_MAX = 2
LOG_STD_MIN = -20


def gaussian_likelihood(x, mu, log_std):
  pre_sum = -0.5 * (
    ((x - mu) / (tf.exp(log_std) + EPS)) ** 2 + 2 * log_std + np.log(
      2 * np.pi))
  return tf.reduce_sum(input_tensor=pre_sum, axis=1)


def apply_squashing_func(mu, pi, logp_pi):
  # Adjustment to log prob
  # NOTE: This formula is a little bit magic. To get an understanding of where it
  # comes from, check out the original SAC paper (arXiv 1801.01290) and look in
  # appendix C. This is a more numerically-stable equivalent to Eq 21.
  # Try deriving it yourself as a (very difficult) exercise. :)
  logp_pi -= tf.reduce_sum(
    input_tensor=2 * (np.log(2) - pi - tf.nn.softplus(-2 * pi)), axis=1)

  # Squash those unbounded actions!
  mu = tf.tanh(mu)
  pi = tf.tanh(pi)
  return mu, pi, logp_pi


def mlp(hidden_sizes, activation):
  model = tf.keras.Sequential()
  for size in hidden_sizes:
    model.add(tf.keras.layers.Dense(size, activation=activation))
  return model


class MlpActor(Model):
  def __init__(self, action_space, hidden_sizes=(256, 256),
               activation=tf.tanh):
    super(MlpActor, self).__init__()
    self.core = mlp(hidden_sizes, activation)
    self.head_mu = tf.keras.layers.Dense(action_space.shape[0])
    self.head_log_std = tf.keras.layers.Dense(action_space.shape[0])
    self.action_space = action_space

  def call(self, x):
    x = self.core(x)
    mu = self.head_mu(x)
    log_std = self.head_log_std(x)

    log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)
    std = tf.exp(log_std)
    pi = mu + tf.random.normal(tf.shape(input=mu)) * std
    logp_pi = gaussian_likelihood(pi, mu, log_std)

    mu, pi, logp_pi = apply_squashing_func(mu, pi, logp_pi)

    # Make sure actions are in correct range
    action_scale = self.action_space.high[0]
    mu *= action_scale
    pi *= action_scale

    return mu, pi, logp_pi


class MlpCritic(Model):
  def __init__(self, input_dim, hidden_sizes=(256, 256),
               activation=tf.tanh):
    super(MlpCritic, self).__init__()
    self.net = mlp(hidden_sizes, activation)
    self.net.add(tf.keras.layers.Dense(1))
    # Thanks to build, model weights are created right away.
    self.net.build((None, input_dim))

  def call(self, x, a):
    x = self.net(tf.concat([x, a], axis=-1))
    x = tf.squeeze(x, axis=1)
    return x
