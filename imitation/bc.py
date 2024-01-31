import os
import time
import random 

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary as th_summary
from tqdm import tqdm

# Dataset utils
from torch.utils.data import IterableDataset, DataLoader

# Robohive dependencies
import gym
from robohive.logger.grouped_datasets import Trace as RoboHive_Trace

# Config and logging helpers
import tools
from configurator import generate_args, get_arg_dict
from th_logger import TBXLogger as TBLogger

class BCIterableDataset(IterableDataset):
  def __init__(self, dataset_path, scale_obs=False, seed=111):
    self.seed = seed
    self.dataset_path = dataset_path
    self.scale_obs = scale_obs # From original range to [-1,1] by default

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
    # Recover the min, max for the observations
    self.obs_min, self.obs_max = \
      self.buffer["observations"].min(), self.buffer["observations"].max()

    # DEBUG
    for k in ["observations", "actions", "dones", "target_positions"]:
      print(f" # DBG: Buffer {k} shape: {np.shape(self.buffer[k])}")
      print(f" # DBG: Data range of {k}: {self.buffer[k].min(), self.buffer[k].max()}")
    
    print(f"\nInitialized IterDataset with {len(self.ep_filenames)} episodes, totalling {self.buffer_length} steps.\n")
  def __iter__(self):
    while True:
      idx = th.randint(0, self.buffer_length, [1])

      # Scaling observation and target_positions
      obs_list = self.buffer["observations"][idx].astype(np.float32)
      target_pos_list = self.buffer["target_positions"][idx].astype(np.float32)
      if self.scale_obs:
        obs_list = self.normalize_observation(obs_list)
        target_pos_list = self.normalize_observation(target_pos_list)

      # observation, action, done, target_pos of a random step from the buffer
      yield obs_list, \
            self.buffer["actions"][idx].astype(np.float32), \
            self.buffer["dones"][idx].astype(np.float32), \
            target_pos_list  
  
  # Used for observation normalization
  @staticmethod
  def _scale_field(a, old_min=0., old_max=1., new_min=-1, new_max=1.):
    assert old_min < old_max, f"Invalid scaling: old_min {old_min} >= old_max: {old_max}"
    assert new_min < new_max, f"Invalid scaling: new_min {new_min} >= new_max: {new_max}"
    return ((a - old_min) * (new_max - new_min)) / (old_max - old_min) + new_min

  def normalize_observation(self, x):
    return self._scale_field(x, old_min=self.obs_min, old_max=self.obs_max)

def make_dataloader(dataset_path, batch_size, scale_obs=False, seed=111, num_workers=2):
  def worker_init_fn(worker_id):
    # worker_seed = th.initial_seed() % (2 ** 32)
    worker_seed = 133754134 + worker_id

    random.seed(worker_seed)
    np.random.seed(worker_seed)

  th_seed_gen = th.Generator()
  th_seed_gen.manual_seed(133754134 + seed)

  dloader = iter(
    DataLoader(
      BCIterableDataset(dataset_path=dataset_path, scale_obs=scale_obs),
        batch_size=batch_size, num_workers=num_workers,
        worker_init_fn=worker_init_fn, generator=th_seed_gen)
  )

  return dloader

# Eval helper
def eval_agent(env, agent, args, dataset=None):
  """
    args: used to recover eval settings, and observation scaling
    dataset: used to scale observations
  """
  solved_list = []
  video_dict = {}
  n_video_saved = 0

  for eval_ep_idx in range(args.eval_n_episodes):
    obs = env.reset()
    target_pos = env.get_target_pos()
    obs_target = np.concatenate([obs, target_pos])
    if args.scale_obs:
      obs_target = dataset.normalize_observation(obs_target)
    
    solved = False
    MAX_STEPS=500
    t = 0
    
    if args.save_videos and n_video_saved < args.save_videos_n_max:
      # NOTE: video collectin and saving is expensive process
      ep_video_data = []

    while not solved and t < MAX_STEPS:
      with th.no_grad():
        action = agent(th.Tensor(obs_target)[None, :].float().to(agent.device))
      action = action[0].cpu().numpy()

      obs, _, _, info = env.step(action)
      solved = info["solved"]
      obs_target = np.concatenate([obs, target_pos])
      if args.scale_obs:
        obs_target = dataset.normalize_observation(obs_target)

      if args.save_videos and n_video_saved < args.save_videos_n_max:
        ep_video_data.append(env.get_visuals()["rgb:front_cam:240x424:2d"])

      t += 1
    
    # Cummulate stats
    solved_list.append(solved)
    if args.save_videos and n_video_saved < args.save_videos_n_max:
      video_dict[f"eval_ep_{eval_ep_idx}"] = \
        np.array(ep_video_data)[None, :].transpose(0, 1, 4, 2, 3)
      n_video_saved += 1

  return solved_list, video_dict

# Agent models
# TODO: separate to models.py in case we have more models
class DeterministicActor(nn.Module):
  def __init__(self,
                input_dim,
                output_dim,
                n_layers,
                hid_size,
                act_fn=nn.ReLU,
                out_act_fn=nn.Identity):
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

  def to(self, device):
    super().to(device)
    self.device = device
    return self

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
    get_arg_dict("scale-obs", bool, True, metatype="bool"), # Scales obs to [-1,1] range

    ## Actor network params
    get_arg_dict("actor-type", str, "deter", metatype="choice",
      choices=["deter"]),
    get_arg_dict("actor-hid-layers", int, 3),
    get_arg_dict("actor-hid-size", int, 512),

    # Eval protocol
    # TODO: max horizon for the eval step, etc...
    get_arg_dict("eval", bool, True, metatype="bool"),
    get_arg_dict("eval-every", int, int(5e4)), # Every X updates
    get_arg_dict("eval-n-episodes", int, 8),

    # Logging params
    get_arg_dict("save-videos", bool, True, metatype="bool"),
    get_arg_dict("save-videos-n-max", int, 1), # Max number of videos to log
    get_arg_dict("save-model", bool, True, metatype="bool"),
    get_arg_dict("save-model-every", int, int(5e4)), # Every X frames || steps sampled
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

  # Load the dataset
  dataloader = make_dataloader(args.dataset_path, 
                              args.batch_size,
                              scale_obs=args.scale_obs)

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
    env = gym.make("rpFrankaPickPlaceData-v0", 
                  randomize=True)
  else:
    # TODO: create place holder observation and action spaces
    pass

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
  optimizer = th.optim.Adam(agent.parameters(),
                            lr=args.lr,
                            eps=1e-5,
                            weight_decay=args.optim_wd)

  # Training start
  start_time = time.time()
  # Log the number of parameters of the model
  tblogger.log_stats({
      "n_params": agent.get_n_params()
  }, 0, "info")

  # Training loop
  for global_step in (pbar := tqdm(range(0, args.total_steps + args.batch_size, args.batch_size))):
    obs_list, act_list, done_list, target_pos_list = \
      [b.to(device) for b in next(dataloader)]
    
    optimizer.zero_grad()

    obs_target_pos_list = th.cat([
      obs_list, target_pos_list], dim=1)

    actions = agent(obs_target_pos_list)

    bc_loss = F.mse_loss(actions, act_list)
    bc_loss.backward()

    optimizer.step()

    if should_log_training_stats(global_step):
      # print(f"Step {global_step} / {args.total_steps}")
      # print(f"  bc_loss: {bc_loss.item(): 0.3f}")
      pbar.set_description(f"BC Loss: {global_step:7d}/{args.total_steps:7d}: {bc_loss.item():.4f}")

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
      eval_solved_list, eval_video_dict = eval_agent(env, agent, args, dataset=dataloader._dataset)
      tblogger.log_stats({
        "success_rate": np.mean(eval_solved_list),
      }, global_step, prefix="eval")
      for k, v in eval_video_dict.items():
        tblogger.log_video(k, v, global_step, fps=24, prefix="video")

    if args.save_model and should_save_model(global_step):
      model_save_dir = tblogger.get_models_savedir()
      model_save_name = f"agent.{global_step}.ckpt.pth"
      model_save_fullpath = os.path.join(model_save_dir, model_save_name)

      th.save(agent.state_dict(), model_save_fullpath)

  # Clean up
  tblogger.close()
  if args.eval:
    env.close()

if __name__ == "__main__":
  main()
