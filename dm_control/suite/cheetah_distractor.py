# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Cheetah Domain."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import random

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils import rewards


# How long the simulation will run, in seconds.
_DEFAULT_TIME_LIMIT = 10

# Running speed above which reward is 1.
_RUN_SPEED = 10

SUITE = containers.TaggedTasks()


def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  return common.read_model('cheetah_distractor.xml'), common.ASSETS


@SUITE.add()
def run(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the run task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Cheetah(random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit,
                             **environment_kwargs)

@SUITE.add()
def run_linear(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the run task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Cheetah(random=random, distractor_style=1)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit,
                             **environment_kwargs)


class Physics(mujoco.Physics):
  """Physics simulation with additional features for the Cheetah domain."""

  def speed(self):
    """Returns the horizontal speed of the Cheetah."""
    return self.named.data.sensordata['torso_subtreelinvel'][0]


class Cheetah(base.Task):
  """A `Task` to train a running Cheetah."""

  def __init__(self, random=random, distractor_style=0):
    self._distractor_style = distractor_style
    self._step_size = 0.5
    self.sample_new_dir()
    self.x1 = np.random.uniform(-1, 1)
    self.x2 = np.random.uniform(-1, 1)
    super(Cheetah, self).__init__(random=random)

  def initialize_episode(self, physics):
    """Sets the state of the environment at the start of each episode."""
    # The indexing below assumes that all joints have a single DOF.
    assert physics.model.nq == physics.model.njnt
    is_limited = physics.model.jnt_limited == 1
    lower, upper = physics.model.jnt_range[is_limited].T
    physics.data.qpos[is_limited] = self.random.uniform(lower, upper)

    # Stabilize the model before the actual simulation.
    for _ in range(200):
      physics.step()

    physics.data.time = 0
    self._timeout_progress = 0

    cheetah_x = physics.data.qpos[0]

    physics.named.data.qpos['dis1x'] = cheetah_x + self.x1
    physics.named.data.qpos['dis1y'] = np.random.uniform(1, 2)
    physics.named.data.qpos['dis2x'] = cheetah_x + self.x2
    physics.named.data.qpos['dis2y'] = np.random.uniform(1, 2)

    super(Cheetah, self).initialize_episode(physics)

  def sample_new_dir(self):
    dirs = np.random.uniform(-1, 1, size=(2, 2))
    dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8)
    self._current_dir = dirs * self._step_size

  def get_observation(self, physics):
    """Returns an observation of the state, ignoring horizontal position."""
    obs = collections.OrderedDict()
    # Ignores horizontal position to maintain translational invariance.
    obs['position'] = physics.data.qpos[1:].copy()
    obs['velocity'] = physics.velocity()

    cheetah_x = physics.data.qpos[0]
    if self._distractor_style == 0:
      physics.named.data.qpos['dis1x'] = cheetah_x + np.random.uniform(-2, 2)
      physics.named.data.qpos['dis1y'] = np.random.uniform(0, 3)
      physics.named.data.qpos['dis2x'] = cheetah_x + np.random.uniform(-2, 2)
      physics.named.data.qpos['dis2y'] = np.random.uniform(0, 3)
    elif self._distractor_style == 1:
      if random.random() < 0.15:
        self.sample_new_dir()
      self.x1 = np.clip(self.x1 + self._current_dir[0, 0], -3, 3)
      self.x2 = np.clip(self.x2 + self._current_dir[1, 0], -3, 3)
      physics.named.data.qpos['dis1x'] = cheetah_x + self.x1
      physics.named.data.qpos['dis1y'] = np.clip(physics.named.data.qpos['dis1y'] + self._current_dir[0, 1], 0, 3)
      physics.named.data.qpos['dis2x'] = cheetah_x + self.x2
      physics.named.data.qpos['dis2y'] = np.clip(physics.named.data.qpos['dis2y'] + self._current_dir[1, 1], 0, 3)

    return obs

  def get_reward(self, physics):
    """Returns a reward to the agent."""
    return rewards.tolerance(physics.speed(),
                             bounds=(_RUN_SPEED, float('inf')),
                             margin=_RUN_SPEED,
                             value_at_margin=0,
                             sigmoid='linear')
