<!-- =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= -->

<!-- # RoboHive -->

![PyPI](https://img.shields.io/pypi/v/robohive)
![PyPI - License](https://img.shields.io/pypi/l/robohive)
[![Downloads](https://pepy.tech/badge/robohive)](https://pepy.tech/project/robohive)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rdSgnsfUaE-eFLjAkFHeqfUWzAK8ruTs?usp=sharing)
[![Slack](https://img.shields.io/badge/Slack-4A154B?style=for-the-badge&logo=slack&logoColor=white)](https://robohiveworkspace.slack.com)
[![Documentation](https://img.shields.io/static/v1?label=Wiki&message=Documentation&color=<green)](https://github.com/vikashplus/robohive/wiki)

![RoboHive Social Preview](https://github.com/vikashplus/robohive/assets/12837145/04aff6da-f9fa-4f5f-abc6-cfcd70c6cd90)
`RoboHive` is a collection of environments/tasks simulated with the [MuJoCo](http://www.mujoco.org/) physics engine exposed using the OpenAI-Gym API.

# Getting Started
   Getting started with RoboHive is as simple as -
   ``` bash
   # Install RoboHive
   pip install robohive
   # Initialize RoboHive
   robohive_init
   # Demo an environment
   python -m robohive.utils.examine_env -e FrankaReachRandom-v0
   ```

   or, alternatively for editable installation -

   ``` bash
   # Clone RoboHive
   git clone --recursive https://github.com/vikashplus/robohive.git; cd robohive
   # Install (editable) RoboHive
   pip install -e .
   # Demo an environment
   python -m robohive.utils.examine_env -e FrankaReachRandom-v0
   ```

   See [detailed installation instructions](./setup/README.md) for options on mujoco-python-bindings and  visual-encoders ([R3M](https://sites.google.com/view/robot-r3m/), [RRL](https://sites.google.com/view/abstractions4rl), [VC](https://eai-vc.github.io/)), and [frequently asked questions](https://github.com/vikashplus/robohive/wiki/6.-Tutorials-&-FAQs#installation) for more details.

# Suites
*RoboHive* contains a variety of environments, which are organized as suites. Each suite is a collection of loosely related environments. The following suites are provided at the moment with plans to improve the diversity of the collection.

**Hand-Manipulation-Suite** [(video)](https://youtu.be/jJtBll8l_OM)
:-------------------------:
![Alt text](https://raw.githubusercontent.com/vikashplus/robohive/f786982204e85b79bd921aa54ffebf3a7887de3d/mj_envs/hand_manipulation_suite/assets/tasks.jpg?raw=false "Hand Manipulation Suite") A collection of environments centered around dexterous manipulation. Standard ADROIT benchmarks introduced in [Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations, RSS2018](https://sites.google.com/corp/view/deeprl-dexterous-manipulation).) are a part of this suite


Arm-Manipulation-Suite
:-------------------------:
![Alt text](https://github.com/vikashplus/robohive/assets/12837145/ef072b90-42e7-414b-9da0-45c87c31443a?raw=false "Arm Manipulation Suite") A collection of environments centered around Arm manipulation.


Myo-Suite [(website)](https://sites.google.com/view/myosuite)
:-------------------------:
![Alt text](https://github.com/vikashplus/robohive/assets/12837145/0db70854-cb90-4360-8bd9-42cd1b5446c1?raw=false "Myo_Suite") A collection of environments centered around Musculoskeletal control.


Myo/MyoDM-Suite [(Website)](https://sites.google.com/view/myodex)
:-------------------------:
![myodm_task_suite](https://github.com/vikashplus/robohive/assets/12837145/2ca62e77-6827-4029-930e-b95ab86ae0f4) A collection of musculoskeletal environments for dexterous manipulation introduced as MyoDM in [MyoDeX](https://sites.google.com/view/myodex).


MultiTask Suite
:-------------------------:
![Alt text](https://github.com/vikashplus/robohive/assets/12837145/b7f314b9-8d4e-4e58-b791-6df774b91d21?raw=false "Myo_Suite") A collection of environments centered around multi-task. Standard [RelayKitchen benchmarks](https://relay-policy-learning.github.io/) are a part of this suite.

## - TCDM Suite (WIP)
   This suite contains a collection of environments centered around dexterous manipulation. Standard [TCDM benchmarks](https://pregrasps.github.io/) are a part of this suite

## - ROBEL Suite (Coming soon)
   This suite contains a collection of environments centered around real-world locomotion and manipulation. Standard [ROBEL benchmarks](http://roboticsbenchmarks.org/) are a part of this suite

## Multi-Robot Environment

### Virtual environment
We use Anaconda as virtual environment manager for Ubuntu 20.04 LTS
The environment is created with:
```bash
conda create -n robohive python=3.10
conda activate robohive
```
- Then, perform the setings in the "Getting Started" section above for the editable install, but using the `multi-robot` branch.
- 4 Franka Arms for cube triage task env name: `FrankaReachFixedMulti-v0`
- Continuous joint action space over all robots of shape `Box(9 x N_ROBOTS)`
- Getting the env up and recovering visual inputs with `env.get_visuals()`

```python
import gym
import robohive

env = gym.make("FrankaReachFixedMulti-v0")
env.reset()

visuals = env.get_visuals()
# Expected output:
"""
{
      'time': array([0.]),
      'rgb:franka0_front_cam:256x256:1d': array([122, 118, 108, ...,  92,  88,  85], dtype=uint8),
      'rgb:franka1_front_cam:256x256:1d': array([138, 130, 121, ...,  80,  77,  74], dtype=uint8),
      'rgb:franka2_front_cam:256x256:1d': array([145, 134, 124, ...,  84,  82,  79], dtype=uint8),
      'rgb:franka3_front_cam:256x256:1d': array([149, 139, 129, ...,  80,  78,  75], dtype=uint8)}
"""
```

- Running the environment visualizer (requires display)
```bash
# Fixed robots
DISPLAY=:1 python -m robohive.utils.examine_env_multi -e FrankaReachFixedMulti-v0 --num_episodes=10000
```
```bash
# Random policy robots
DISPLAY=:1 python -m robohive.utils.examine_env -e FrankaReachFixedMulti-v0 --num_episodes=10000
```
- Additional dependencies to get gamepad based tele operation work (WIP)
```bash
# Either install https://github.com/vikashplus/vtils or a local editable fork of the project
# Xbox 360 gamepad support was added into branch `x360-gamepad` of [Rousslan's fork](https://github.com/dosssman/vtils.git)
# Install directly with:
pip install -e git+https://github.com/dosssman/vtils.git@x360-gamepad

# Additionally
pip install inputs # for gamepad support
```

### Tele-operation and data collection

A custom, simplified data collection environment was added as: `rpFrankaRobotiqData-v0`.

**Keyboard teleop**:
```bash
DISPLAY=:1 python robohive/tutorials/ee_teleop_multi.py

# Controls:
# - ESDF akin to WASD for games (front,left,back, right) assuming standing behind the robot
# - Q to raise the end-effector, Z to lower it
# - R to open the gripper, V to close it
# - Arrow keys to rotate the arm, , and . to rotate the end effector.
```

**Gamepad teleop**:
```bash
DISPLAY=:1 python robohive/tutorials/ee_teleop7_multi.py -i gamepad
DISPLAY=:1 python robohive/tutorials/ee_teleop_multi.py -i gamepad -ea "{'randomiz
e':True}" # randomizes start position of objects in the env.
## Data collection for initial BC training
DISPLAY=:1 python robohive/tutorials/ee_teleop_multi.py -i gamepad -ea "{'randomize':True}" -o "data/2024-01-30-dataset/teleop_gamepad_traj_X.h5
# Controls
# - Left stick controls X Y axis movement
# - LT: raise
# - RT: lower
# - Y: close gripper
# - B: open gripper
# - Dpad, LB, RB: end effector rotation
```

Once the data is collected, simultaneously press SELECT+START to stop the data collection and wait for the file to be saved.
The tele-operation script can then be close with `Ctrl+C`.

### Loading and structure of the tele-operation collected data:
```
import numpy as np

from robohive.logger.roboset_logger import RoboSet_Trace
from robohive.logger.grouped_datasets import Trace as RoboHive_Trace

trace0 = RoboHive_Trace("TeleOp Trajectories")
trace0 = trace0.load("data/2024-01-30-pick-place-dataset/teleop_gamepad_traj_00.h5"); trace0

# Output as follows:

Trace_name: dict_keys(['data/2024-01-30-dataset/teleop_gamepad_traj_7'])
<HDF5 group "/Trial0" (8 members)>
	<HDF5 dataset "actions": shape (337, 9), type "<f2">
	<HDF5 dataset "done": shape (337,), type "|b1">
	<HDF5 group "/Trial0/env_infos" (9 members)>
	<HDF5 dataset "observations": shape (337, 37), type "<f2">
	<HDF5 dataset "rewards": shape (337,), type "<f2">
	<HDF5 dataset "target_pos": shape (337, 3), type "<f2">
	<HDF5 dataset "time": shape (337,), type "<f2">
	<HDF5 group "/Trial0/visual_obs" (9 members)>

# Loading the relevant fields for IL as an example:
# Might want to turn that to numpy array's later on.
observations, actions, done, target_pos = \
  trace["Trial0"].get("observations"), \
  trace["Trial0"].get("actions"), \
  trace["Trial0"].get("done"), \
  trace["Trial0"].get("target_pos")
```

### Imitation Learning

Additional dependencies
```
pip install wandb==0.16.2 tensorboardx==2.6.2.2 tensorboard==2.15.1 nvsmi==0.4.2 torchinfo==1.8.0
```

# Citation
If you find `RoboHive` useful in your research,
- please consider supporting the project by providing a [star ⭐](https://github.com/vikashplus/robohive/stargazers)
- please consider citing our project by using the following BibTeX entry:



```bibtex
@Misc{RoboHive2020,
  title = {RoboHive -- A Unified Framework for Robot Learning},
  howpublished = {\url{https://sites.google.com/view/robohive}},
  year = {2020},
  url = {https://sites.google.com/view/robohive},
}
