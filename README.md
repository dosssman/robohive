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
- Teleoperation is WIP
- Running the environment visualizer (requires display)
```bash
# Fixed robots
DISPLAY=:1 python -m robohive.utils.examine_env_multi -e FrankaReachFixedMulti-v0 --num_episodes=10000
```
```bash
# Random policy robots
DISPLAY=:1 python -m robohive.utils.examine_env -e FrankaReachFixedMulti-v0 --num_episodes=10000
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
