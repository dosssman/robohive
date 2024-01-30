import os
import time
import random 

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary as th_summary

# Dataset utils
from torch.utils.data import IterableDataset, DataLoader

# Robohive dependencies
from robohive.logger.grouped_datasets import Trace as RoboHive_Trace

# Config and logging helpers
import tools
from configurator import generate_args, get_arg_dict
from th_logger import TBXLogger as TBLogger

# There is no empty step in this batch
class BCIterableDataset(IterableDataset):
  def __init__(self, dataset_path, seed=111):
    self.seed = seed
    self.dataset_path = dataset_path

    # Read episode filenames in the dataset path
    self.ep_filenames = os.listdir(dataset_path)
    # NOTE: buffering all trajectories might not be sustainable for larger datasets
    # Consider lazy loading scheme instead
    self.buffer = {
      "observations": [],
      "actions": [],
      "dones": [],
      "target_positions": []
    }

    for ep_filename in self.ep_filenames:
      ep_fullpath = os.path.join(self.dataset_path, ep_filename)

      # Read the Robohive trace
      trace = RoboHive_Trace("")
      trace = trace.load(ep_fullpath)

      ep_observations, ep_actions, ep_dones, ep_target_positions = \
        trace["Trial0"].get("observations"), \
        trace["Trial0"].get("actions"), \
        trace["Trial0"].get("done"), \
        trace["Trial0"].get("target_pos"), \
      
      self.buffer["observations"].append(ep_observations)
      self.buffer["actions"].append(ep_actions)
      self.buffer["dones"].append(ep_dones)
      self.buffer["target_positions"].append(ep_target_positions)
    
    for k, v in self.buffer.items():
      self.buffer[k] = np.concatenate(v)

    # Adjusting shapes
    self.buffer["dones"] = self.buffer["dones"][:, None]

    # Recover total sample number in the buffer
    self.buffer_length = self.buffer["dones"].shape[0]

    # for k in ["observations", "actions", "dones", "target_positions"]:
    #   print(f"# DBG: Buffer {k} shape: {np.shape(self.buffer[k])}")
    
    print(f"\nInitialized IterDataset with {len(self.ep_filenames)} episodes, totalling {self.buffer_length} steps.\n")
  
  def __iter__(self):
    while True:
      idx = th.randint(0, self.buffer_length, [1])

      # observation, action, done, target_pos of a random step from the buffer
      yield self.buffer["observations"][idx].astype(np.float32), \
            self.buffer["actions"][idx].astype(np.float32), \
            self.buffer["dones"][idx].astype(np.float32), \
            self.buffer["target_positions"][idx].astype(np.float32)

def make_dataloader(dataset_path, batch_size, seed=111, num_workers=2):
  def worker_init_fn(worker_id):
    # worker_seed = th.initial_seed() % (2 ** 32)
    worker_seed = 133754134 + worker_id

    random.seed(worker_seed)
    np.random.seed(worker_seed)

  th_seed_gen = th.Generator()
  th_seed_gen.manual_seed(133754134 + seed)

  dloader = iter(
    DataLoader(
      BCIterableDataset(dataset_path=dataset_path),
        batch_size=batch_size, num_workers=num_workers,
        worker_init_fn=worker_init_fn, generator=th_seed_gen)
  )

  return dloader

def main():
  # region: Generating additional hyparams
  CUSTOM_ARGS = [
    # General hyper parameters
    get_arg_dict("seed", int, 111),
    get_arg_dict("total-steps", int, 500_000),
    
    # Behavior hyparams
    get_arg_dict("dataset-path", str, "../data/2024-01-30-pick-place-dataset/"),
    get_arg_dict("batch-size", int, 32),
    get_arg_dict("lr", float, 2.5e-4), # Learning rate
    get_arg_dict("optim-wd", float, 0), # weight decay for Adam optim
    get_arg_dict("loss-type", str, "mse", metatype="choice",
      choices=["mse"]),

    ## Actor network params
    get_arg_dict("actor-type", str, "deter", metatype="choice",
      choices=["deter"]),
    get_arg_dict("actor-hid-layers", int, 3),
    get_arg_dict("actor-hid-size", int, 512),

    # Eval protocol
    # TODO: max horizon for the eval step, etc...
    get_arg_dict("eval", bool, True, metatype="bool"),
    get_arg_dict("eval-every", int, int(5e3)), # Every X updates
    get_arg_dict("eval-n-episodes", int, 5),

    # Logging params
    get_arg_dict("save-videos", bool, False, metatype="bool"),
    get_arg_dict("save-model", bool, True, metatype="bool"),
    get_arg_dict("save-model-every", int, int(5e3)), # Every X frames || steps sampled
    get_arg_dict("log-training-stats-every", int, int(100)), # Every X model update
    get_arg_dict("logdir-prefix", str, "./logs/") # Overrides the default one
  ]
  args = generate_args(CUSTOM_ARGS)
  # endregion: Generating additional hyparams

  # Seeding
  random.seed(args.seed)
  np.random.seed(args.seed)
  th.manual_seed(args.seed)
  th.cuda.manual_seed_all(args.seed)
  th.backends.cudnn.deterministic = args.torch_deterministic
  # th.backends.cudnn.benchmark = args.cudnn_benchmark

  # Set device as GPU
  device = tools.get_device(args) if (not args.cpu and th.cuda.is_available()) else th.device("cpu")

  # Experiment logger
  tblogger = TBLogger(exp_name=args.exp_name, args=args)
  print(f"# Logdir: {tblogger.logdir}")

  should_log_training_stats = tools.Every(args.log_training_stats_every)
  should_eval = tools.Every(args.eval_every)
  should_save_model = tools.Every(args.save_model_every)

  # Environment instantiation
  if args.eval:
    # TODO: instantiate environment
    pass
  else:
    # TODO: create place holder observation and action spaces
    pass

  # Agent models
  # TODO: separate to models.py in case we have more models
  class DeterministicActor(nn.Module):
    def __init__(self,
                  input_dim,
                  output_dim,
                  n_layers,
                  hid_size,
                  act_fn=nn.ReLU,
                  out_act_fn=nn.Tanh):
      super().__init__()

      network = []

      for h0, h1 in zip(
        [input_dim, *[hid_size for _ in range(n_layers)]],
        [*[hid_size for _ in range(n_layers)], output_dim],
        ):
        network.extend([
          nn.Linear(h0, h1),
          act_fn()])
      
      network.pop()
      network.append(out_act_fn())
      
      self.network = nn.Sequential(*network)

      # TODO: init scehems
    
    def forward(self, x):
      # TODO: some asserts on the type and shape ?
      return self.network(x)

    def get_n_params(self):
      return sum(p.numel() for p in self.parameters())

  # Agent instantiation
  if args.actor_type == "deter":
    agent = DeterministicActor(
      37+3, # TODO soft code
      9,
      n_layers=args.actor_hid_layers,
      hid_size=args.actor_hid_size).to(device)
  else:
    raise NotImplementedError(f"Unsupported agent type: {args.agent_type}")

  # DBG: agent structure
  print(agent)
  th_summary(agent)

  # Optimizers
  # TODO: Add Apex support ?
  optimizer = th.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5, weight_decay=args.optim_wd)

  # Training start
  start_time = time.time()
  # num_updates = args.total_steps // args.batch_size # Total number of updates that will take place in this experiment
  n_updates = 0 # Progressively tracks the number of network updats
  # Log the number of parameters of the model
  tblogger.log_stats({
      "n_params": agent.get_n_params()
  }, 0, "info")

  # Training loop
  for global_step in range(0, args.total_steps, args.batch_size):
    obs_list, act_list, done_list, target_pos_list = \
      [b.to(device) for b in next(dataloader)]
    
    
    optimizer.zero_grad()

    obs_target_pos_list = th.cat([
      obs_list, target_pos_list], dim=1)

    actions = agent(obs_target_pos_list)

    bc_loss = F.mse_loss(actions, act_list)
    print(bc_loss)

    optimizer.step()

    if should_log_training_stats(global_step):
      print(f"Step {global_step} / {args.total_steps}")
      # Training stats
      train_stats = {
        "bc_loss": bc_loss.item()
      }
      tblogger.log_stats(train_stats, global_step, prefix="train")

      # Info stats
      info_stats = {
        "global_step": global_step,
        "duration": time.time() - start_time,
        "fps": tblogger.track_duration("fps", global_step),
        "env_step_duration": tblogger.track_duration("fps_inv", global_step, inverse=True),
      }
      tblogger.log_stats(info_stats, global_step, "info")
    
    if args.eval and should_eval(global_step):
      pass # TODO
    if args.save_model and should_save_model(global_step):
      pass # TODO

    break

  # Clean up
  tblogger.close()
  if args.eval:
    # envs.close()
    pass # TODO