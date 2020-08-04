import gym
from mrunner.helpers.client_helper import get_configuration

from spinup.sac import sac


def get_get_env(env_name):
  def get_env():
    return gym.make(env_name)

  return get_env


def main(task, seed, steps, replay_size, batch_size, hidden_sizes):
  sac(get_get_env(task), seed=seed, steps=steps, replay_size=replay_size,
      batch_size=batch_size, actor_kwargs=dict(hidden_sizes=hidden_sizes),
      critic_kwargs=dict(hidden_sizes=hidden_sizes))


if __name__ == '__main__':
  config = get_configuration(print_diagnostics=True, with_neptune=True)
  experiment_id = config.pop('experiment_id')

  main(**config)
