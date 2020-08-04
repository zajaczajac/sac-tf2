import random
import time

import numpy as np
import tensorflow as tf

from spinup import models
from spinup.replay_buffers import ReplayBuffer
from spinup.utils.logx import EpochLogger


def sac(env_fn, actor_cl=models.MlpActor, actor_kwargs=None,
        critic_cl=models.MlpCritic, critic_kwargs=None, seed=0,
        steps=2000000, log_every=10000, replay_size=1000000, gamma=0.99,
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=256, start_steps=10000,
        update_after=1000, update_every=50, num_test_episodes=1,
        max_ep_len=150, save_freq_epochs=1):
  """
  Non-obvious args:
      polyak (float): Interpolation factor in polyak averaging for target
          networks. Target networks are updated towards main networks
          according to:

          .. math:: \\theta_{\\text{targ}} \\leftarrow
              \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

          where :math:`\\rho` is polyak. (Always between 0 and 1, usually
          close to 1.)

      alpha (float): Entropy regularization coefficient. (Equivalent to
          inverse of reward scale in the original SAC paper.)

      batch_size (int): Minibatch size for SGD.

      start_steps (int): Number of steps for uniform-random action selection,
          before running real policy. Helps exploration.

      update_after (int): Number of env interactions to collect before
          starting to do gradient descent updates. Ensures replay buffer
          is full enough for useful updates.

      update_every (int): Number of env interactions that should elapse
          between gradient descent updates. Note: Regardless of how long
          you wait between updates, the ratio of env steps to gradient steps
          is locked to 1.

      num_test_episodes (int): Number of episodes to test the deterministic
          policy at the end of each epoch.

      max_ep_len (int): Maximum length of trajectory / episode / rollout.

      save_freq_epochs (int): How often (in terms of gap between epochs) to save
          the current policy and value function.
  """

  logger = EpochLogger()
  logger.save_config(locals())

  random.seed(seed)
  tf.random.set_seed(seed)
  np.random.seed(seed)

  env, test_env = env_fn(), env_fn()
  obs_dim = env.observation_space.shape[0]
  act_dim = env.action_space.shape[0]
  # This implementation assumes all dimensions share the same bound!
  assert np.all(env.action_space.high == env.action_space.high[0])

  if actor_kwargs is None:
    actor_kwargs = {}
  if critic_kwargs is None:
    critic_kwargs = {}

  # Share information about action space with policy architecture
  actor_kwargs['action_space'] = env.action_space
  critic_kwargs['input_dim'] = obs_dim + act_dim

  # Experience buffer
  replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim,
                               size=replay_size)

  actor = actor_cl(**actor_kwargs)

  critic1 = critic_cl(**critic_kwargs)
  target_critic1 = critic_cl(**critic_kwargs)
  target_critic1.set_weights(critic1.get_weights())

  critic2 = critic_cl(**critic_kwargs)
  target_critic2 = critic_cl(**critic_kwargs)
  target_critic2.set_weights(critic2.get_weights())

  critic_variables = critic1.trainable_variables + critic2.trainable_variables

  optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

  @tf.function
  def get_action(o, deterministic=tf.constant(False)):
    mu, pi, logp_pi = actor(tf.expand_dims(o, 0))
    if deterministic:
      return mu[0]
    else:
      return pi[0]

  @tf.function
  def learn_on_batch(obs1, obs2, acts, rews, done):
    with tf.GradientTape(persistent=True) as g:
      # Main outputs from computation graph
      mu, pi, logp_pi = actor(obs1)
      q1 = critic1(obs1, acts)
      q2 = critic2(obs1, acts)

      # compose q with pi, for pi-learning
      q1_pi = critic1(obs1, pi)
      q2_pi = critic2(obs1, pi)

      # get actions and log probs of actions for next states, for Q-learning
      _, pi_next, logp_pi_next = actor(obs2)

      # target q values, using actions from *current* policy
      target_q1 = target_critic1(obs2, pi_next)
      target_q2 = target_critic2(obs2, pi_next)

      # Min Double-Q:
      min_q_pi = tf.minimum(q1_pi, q2_pi)
      min_target_q = tf.minimum(target_q1, target_q2)

      # Entropy-regularized Bellman backup for Q functions, using Clipped Double-Q targets
      q_backup = tf.stop_gradient(rews + gamma * (1 - done) * (
        min_target_q - alpha * logp_pi_next))

      # Soft actor-critic losses
      pi_loss = tf.reduce_mean(alpha * logp_pi - min_q_pi)
      q1_loss = 0.5 * tf.reduce_mean((q_backup - q1) ** 2)
      q2_loss = 0.5 * tf.reduce_mean((q_backup - q2) ** 2)
      value_loss = q1_loss + q2_loss

    # Compute gradients and do updates
    actor_gradients = g.gradient(pi_loss, actor.trainable_variables)
    optimizer.apply_gradients(
      zip(actor_gradients, actor.trainable_variables))
    critic_gradients = g.gradient(value_loss, critic_variables)
    optimizer.apply_gradients(
      zip(critic_gradients, critic_variables))
    del g

    # Polyak averaging for target variables
    for v, target_v in zip(critic1.trainable_variables,
                           target_critic1.trainable_variables):
      target_v.assign(polyak * target_v + (1 - polyak) * v)
    for v, target_v in zip(critic2.trainable_variables,
                           target_critic2.trainable_variables):
      target_v.assign(polyak * target_v + (1 - polyak) * v)

    return dict(pi_loss=pi_loss, q1_loss=q1_loss, q2_loss=q2_loss, q1=q1, q2=q2,
                logp_pi=logp_pi)

  def test_agent():
    for j in range(num_test_episodes):
      o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
      while not (d or (ep_len == max_ep_len)):
        # Take deterministic actions at test time
        o, r, d, _ = test_env.step(
          get_action(tf.convert_to_tensor(o), tf.constant(True)))
        ep_ret += r
        ep_len += 1
      logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

  start_time = time.time()
  o, ep_ret, ep_len = env.reset(), 0, 0

  # Main loop: collect experience in env and update/log each epoch
  for t in range(steps):

    # Until start_steps have elapsed, randomly sample actions
    # from a uniform distribution for better exploration. Afterwards,
    # use the learned policy.
    if t > start_steps:
      a = get_action(tf.convert_to_tensor(o))
    else:
      a = env.action_space.sample()

    # Step the env
    o2, r, d, _ = env.step(a)
    ep_ret += r
    ep_len += 1

    # Ignore the "done" signal if it comes from hitting the time
    # horizon (that is, when it's an artificial terminal signal
    # that isn't based on the agent's state)
    d = False if ep_len == max_ep_len else d

    # Store experience to replay buffer
    replay_buffer.store(o, a, r, o2, d)

    # Super critical, easy to overlook step: make sure to update
    # most recent observation!
    o = o2

    # End of trajectory handling
    if d or (ep_len == max_ep_len):
      logger.store(EpRet=ep_ret, EpLen=ep_len)
      o, ep_ret, ep_len = env.reset(), 0, 0

    # Update handling
    if t >= update_after and t % update_every == 0:
      for j in range(update_every):
        batch = replay_buffer.sample_batch(batch_size)
        results = learn_on_batch(**batch)
        logger.store(LossPi=results['pi_loss'],
                     LossQ1=results['q1_loss'],
                     LossQ2=results['q2_loss'],
                     Q1Vals=results['q1'], Q2Vals=results['q2'],
                     LogPi=results['logp_pi'])

    # End of epoch wrap-up
    if ((t + 1) % log_every == 0) or (t + 1 == steps):
      epoch = (t + 1 + log_every - 1) // log_every

      # Save model
      if (epoch % save_freq_epochs == 0) or (t + 1 == steps):
        # TODO: implement saving
        pass

      # Test the performance of the deterministic version of the agent.
      test_agent()

      # Log info about epoch
      logger.log_tabular('Epoch', epoch)
      logger.log_tabular('EpRet', with_min_and_max=True)
      logger.log_tabular('TestEpRet', with_min_and_max=True)
      logger.log_tabular('EpLen', average_only=True)
      logger.log_tabular('TestEpLen', average_only=True)
      logger.log_tabular('TotalEnvInteracts', t + 1)
      logger.log_tabular('Q1Vals', with_min_and_max=True)
      logger.log_tabular('Q2Vals', with_min_and_max=True)
      logger.log_tabular('LogPi', with_min_and_max=True)
      logger.log_tabular('LossPi', average_only=True)
      logger.log_tabular('LossQ1', average_only=True)
      logger.log_tabular('LossQ2', average_only=True)

      logger.log_tabular('Time', time.time() - start_time)
      logger.dump_tabular()
