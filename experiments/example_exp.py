from mrunner.helpers.specification_helper import create_experiments_helper

config = {
  'steps': int(1e6),
  'replay_size': int(1e6),
  'batch_size': 256,
  'hidden_sizes': [256, 256],
}

params_grid = {
  'seed': [1, 2, 3, 4, 5],
  'task': ['Humanoid-v3'],
}
name = globals()['script'][:-3]

experiments_list = create_experiments_helper(
  experiment_name=name,
  project_name=None,
  script='python3 run_example.py',
  python_path='.',
  tags=[name],
  base_config=config,
  params_grid=params_grid)
