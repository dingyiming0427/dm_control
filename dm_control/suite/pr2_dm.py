# Copyright 2017 The dm_control Authors.## Licensed under the Apache License, Version 2.0 (the "License");# you may not use this file except in compliance with the License.# You may obtain a copy of the License at##    http://www.apache.org/licenses/LICENSE-2.0## Unless required by applicable law or agreed to in writing, software# distributed under the License is distributed on an "AS IS" BASIS,# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.# See the License for the specific language governing permissions and# limitations under the License.# ============================================================================"""PR2 Domain."""from __future__ import absolute_importfrom __future__ import divisionfrom __future__ import print_functionimport collectionsimport numpy as npfrom dm_control import mujocofrom dm_control.rl import controlfrom dm_control.suite import basefrom dm_control.utils import containersfrom dm_control.suite import commonfrom dm_control.utils import rewardsimport osfrom dm_control.utils import io as resourcesSUITE = containers.TaggedTasks()_DEFAULT_TIME_LIMIT = 10def get_model_and_assets():  """Returns a tuple containing the model XML string and a dict of assets."""  return resources.GetResource(os.path.join(os.path.dirname(__file__), "pr2_assets", "pr2.xml")), common.ASSETS@SUITE.add('benchmarking')def reach(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):  """Returns the run task."""  physics = mujoco.Physics.from_xml_string(*get_model_and_assets())  task = PR2(random=random)  environment_kwargs = environment_kwargs or {}  return control.Environment(physics, task, time_limit=time_limit,                             **environment_kwargs)class PR2(base.Task):  """A `Task` to train a running Cheetah."""  def initialize_episode(self, physics):    """Sets the state of the environment at the start of each episode."""    # Stabilize the model before the actual simulation.    for _ in range(200):      physics.step()    physics.data.time = 0    self._timeout_progress = 0    super(PR2, self).initialize_episode(physics)  def after_step(self, physics):    physics.data.qacc[7:] += np.random.uniform(-0.1, 0.1, size=physics.data.qacc[7:].shape)    super(PR2, self).after_step(physics)  def get_observation(self, physics):    """Returns an observation of the state, ignoring horizontal position."""    obs = collections.OrderedDict()    # Ignores horizontal position to maintain translational invariance.    obs['position'] = physics.data.qpos[:7].copy()    obs['velocity'] = physics.data.qvel[:7].copy()    obs['ee-goal'] = physics.data.geom_xpos[14] - physics.data.geom_xpos[-1]    return obs  def get_reward(self, physics):    """Returns a reward to the agent."""    reward = -np.linalg.norm(physics.data.geom_xpos[14] - physics.data.geom_xpos[-1])    # ret = rewards.tolerance(reward,    #                          bounds=(-3, 0),    #                          margin=0,    #                          value_at_margin=0,    #                          sigmoid='linear')    # print(reward, ret)    return rewardif __name__ == '__main__':  # Load one task:  env = run()  # Step through an episode and print out reward, discount and observation.  action_spec = env.action_spec()  time_step = env.reset()  while not time_step.last():    action = np.random.uniform(action_spec.minimum,                               action_spec.maximum,                               size=action_spec.shape)    time_step = env.step(action)    print(time_step.reward, time_step.discount, time_step.observation)