import argparse
import os

from utils.compat import apply_runtime_compat_patches, ensure_d4rl_registered


def main():
    os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")
    parser = argparse.ArgumentParser(description="Smoke test for d4rl/gym compatibility patches")
    parser.add_argument("--env-id", default="halfcheetah-medium-v2")
    args = parser.parse_args()

    patch_results = apply_runtime_compat_patches()
    print("Patch results:", patch_results)

    import gym
    ensure_d4rl_registered(args.env_id)

    env = gym.make(args.env_id)
    obs, info = env.reset()
    action = env.action_space.sample()
    step_out = env.step(action)
    print("reset_obs_shape:", getattr(obs, "shape", None))
    print("step_tuple_len:", len(step_out))
    print("smoke_test:", "ok")


if __name__ == "__main__":
    main()
