import sys
import types
import unittest
from contextlib import contextmanager

from utils.compat import apply_runtime_compat_patches


@contextmanager
def temporary_modules(module_map):
    old_modules = {}
    for name, module in module_map.items():
        old_modules[name] = sys.modules.get(name)
        sys.modules[name] = module

    try:
        yield
    finally:
        for name in module_map:
            previous = old_modules[name]
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


class CompatPatchTests(unittest.TestCase):
    def test_numpy_bool8_patch(self):
        numpy_module = types.ModuleType("numpy")
        numpy_module.bool_ = bool
        if hasattr(numpy_module, "bool8"):
            delattr(numpy_module, "bool8")

        with temporary_modules({"numpy": numpy_module}):
            apply_runtime_compat_patches()
            self.assertTrue(hasattr(numpy_module, "bool8"))
            self.assertIs(numpy_module.bool8, numpy_module.bool_)

    def test_registry_and_time_limit_patch(self):
        gym_module = types.ModuleType("gym")
        envs_module = types.ModuleType("gym.envs")
        wrappers_module = types.ModuleType("gym.wrappers")

        registration_module = types.ModuleType("gym.envs.registration")
        registration_module.registry = {}

        class DummyEnv:
            def step(self, action):
                return 1, 2.0, False, {"x": 1}

        class TimeLimit:
            def __init__(self):
                self.env = DummyEnv()
                self._elapsed_steps = 0
                self._max_episode_steps = 1

            def step(self, action):
                return self.env.step(action)

        time_limit_module = types.ModuleType("gym.wrappers.time_limit")
        time_limit_module.TimeLimit = TimeLimit

        module_map = {
            "gym": gym_module,
            "gym.envs": envs_module,
            "gym.envs.registration": registration_module,
            "gym.wrappers": wrappers_module,
            "gym.wrappers.time_limit": time_limit_module,
        }

        with temporary_modules(module_map):
            apply_runtime_compat_patches()

            self.assertTrue(hasattr(registration_module.registry, "env_specs"))
            self.assertIs(registration_module.registry.env_specs, registration_module.registry)

            tl = TimeLimit()
            output = tl.step(None)
            self.assertEqual(len(output), 5)
            self.assertTrue(output[3])

    def test_normalized_box_env_patch(self):
        d4rl_module = types.ModuleType("d4rl")
        d4rl_utils_module = types.ModuleType("d4rl.utils")
        wrappers_module = types.ModuleType("d4rl.utils.wrappers")

        class Space:
            low = -1.0
            high = 1.0

        class WrappedEnv:
            action_space = Space()

            def step(self, action):
                return "obs", 1.0, False, True, {"ok": True}

        class NormalizedBoxEnv:
            def __init__(self):
                self._wrapped_env = WrappedEnv()
                self._should_normalize = False
                self._reward_scale = 2.0

            def step(self, action):
                # Simulates old d4rl wrapper logic that fails under gym 0.26
                wrapped_step = self._wrapped_env.step(action)
                obs, reward, done, info = wrapped_step
                return obs, reward * self._reward_scale, done, info

        wrappers_module.NormalizedBoxEnv = NormalizedBoxEnv

        module_map = {
            "d4rl": d4rl_module,
            "d4rl.utils": d4rl_utils_module,
            "d4rl.utils.wrappers": wrappers_module,
        }

        with temporary_modules(module_map):
            apply_runtime_compat_patches()
            output = NormalizedBoxEnv().step(None)
            self.assertEqual(len(output), 5)
            self.assertTrue(output[3])
            self.assertEqual(output[1], 2.0)

    def test_ant_env_patch(self):
        gym_module = types.ModuleType("gym")
        spaces_module = types.ModuleType("gym.spaces")
        envs_module = types.ModuleType("gym.envs")
        mujoco_module = types.ModuleType("gym.envs.mujoco")
        mujoco_env_module = types.ModuleType("gym.envs.mujoco.mujoco_env")

        class Box:
            def __init__(self, low, high, shape, dtype):
                self.low = low
                self.high = high
                self.shape = shape
                self.dtype = dtype

        class MujocoEnv:
            def __init__(self, model_path, frame_skip, observation_space):
                self.model_path = model_path
                self.frame_skip = frame_skip
                self.observation_space = observation_space

        spaces_module.Box = Box
        mujoco_env_module.MujocoEnv = MujocoEnv

        d4rl_module = types.ModuleType("d4rl")
        locomotion_module = types.ModuleType("d4rl.locomotion")
        ant_module = types.ModuleType("d4rl.locomotion.ant")

        class FakeRNG:
            def standard_normal(self, size=None):
                return ("standard_normal", size)

        class AntEnv:
            def __init__(self):
                self.np_random = FakeRNG()
                self.data = object()
                MujocoEnv.__init__(self, "ant.xml", 5)

            def reset_model(self):
                return self.np_random.randn(3)

        ant_module.AntEnv = AntEnv

        module_map = {
            "gym": gym_module,
            "gym.spaces": spaces_module,
            "gym.envs": envs_module,
            "gym.envs.mujoco": mujoco_module,
            "gym.envs.mujoco.mujoco_env": mujoco_env_module,
            "d4rl": d4rl_module,
            "d4rl.locomotion": locomotion_module,
            "d4rl.locomotion.ant": ant_module,
        }

        with temporary_modules(module_map):
            apply_runtime_compat_patches()
            env = AntEnv()
            self.assertEqual(env.observation_space.shape, (29,))
            self.assertIs(env.sim.data, env.data)
            self.assertIs(env.physics.data, env.data)
            self.assertEqual(env.reset_model(), ("standard_normal", (3,)))


if __name__ == "__main__":
    unittest.main()
