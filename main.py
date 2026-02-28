import argparse
import pprint
import time
import uuid

from utils.compat import apply_runtime_compat_patches, ensure_d4rl_registered

apply_runtime_compat_patches()


DEFAULT_ENV_ID = "halfcheetah-medium-v2"
DEFAULT_GAMMA = 0.99
DEFAULT_NUM_ENV_STEPS = 1000001
DEFAULT_ONLINE_STEPS = 250001
DEFAULT_EVAL_COUNTER = 10000
DEFAULT_NUM_EVALS = 10
DEFAULT_NORMALISE_STATE = True
DEFAULT_DEP_TARG = True

ALGO_CHOICES = ("td3_n", "bc", "combined")


def load_runtime_dependencies():
    try:
        import gym
        import numpy as np
        import torch
    except ModuleNotFoundError as exc:
        pkg = exc.name or "unknown"
        raise SystemExit(
            f"Missing dependency '{pkg}'. Run setup first (e.g. ./setup.sh), "
            "or install dependencies from environment.yml."
        ) from exc

    from execution_scripts import bc_offline, combined, td3_n_offline
    from utils.plotting_scripts import plot_online_return, plot_online_std

    algo_runners = {
        "td3_n": td3_n_offline,
        "bc": bc_offline,
        "combined": combined,
    }
    plot_functions = {
        "return": plot_online_return,
        "std": plot_online_std,
    }
    return gym, np, torch, algo_runners, plot_functions


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run offline/online policy-switching experiments."
    )
    parser.add_argument(
        "--algo",
        choices=sorted(ALGO_CHOICES),
        default="td3_n",
        help="Algorithm to run.",
    )
    parser.add_argument("--env-id", default=DEFAULT_ENV_ID, help="Gym environment id.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device (default: cuda:0 if available else cpu).",
    )
    parser.add_argument("--n-processors", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--num-env-steps", type=int, default=DEFAULT_NUM_ENV_STEPS)
    parser.add_argument("--online-steps", type=int, default=DEFAULT_ONLINE_STEPS)
    parser.add_argument("--num-evals", type=int, default=DEFAULT_NUM_EVALS)
    parser.add_argument("--eval-counter", type=int, default=DEFAULT_EVAL_COUNTER)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--mem-size", type=int, default=2 ** 20)
    parser.add_argument("--ensemble-num", type=int, default=1)
    parser.add_argument("--critic-factor", type=int, default=10)
    parser.add_argument("--wandb-project", default="Policy Stitching")

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--offline", dest="offline", action="store_true", default=True
    )
    mode_group.add_argument("--online", dest="offline", action="store_false")

    gaussian_group = parser.add_mutually_exclusive_group()
    gaussian_group.add_argument(
        "--gaussian-bc", dest="gaussian_bc", action="store_true", default=True
    )
    gaussian_group.add_argument("--vanilla-bc", dest="gaussian_bc", action="store_false")

    normalise_group = parser.add_mutually_exclusive_group()
    normalise_group.add_argument(
        "--normalise-state",
        dest="normalise_state",
        action="store_true",
        default=DEFAULT_NORMALISE_STATE,
    )
    normalise_group.add_argument(
        "--no-normalise-state", dest="normalise_state", action="store_false"
    )

    dep_targ_group = parser.add_mutually_exclusive_group()
    dep_targ_group.add_argument(
        "--dep-targ", dest="dep_targ", action="store_true", default=DEFAULT_DEP_TARG
    )
    dep_targ_group.add_argument("--no-dep-targ", dest="dep_targ", action="store_false")

    stitch_group = parser.add_mutually_exclusive_group()
    stitch_group.add_argument(
        "--policy-stitch", dest="policy_stitch", action="store_true", default=True
    )
    stitch_group.add_argument(
        "--no-policy-stitch", dest="policy_stitch", action="store_false"
    )

    parser.add_argument("--train-model", action="store_true", default=False)
    parser.add_argument(
        "--plot",
        choices=("none", "return", "std"),
        default="none",
        help="Optional plotting utility to run after training.",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print resolved config before running.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and print config only; do not run training.",
    )
    return parser.parse_args()


def _parse_task_quality(env_id):
    parts = env_id.split("-")
    if len(parts) >= 3:
        task = parts[0]
        data_quality = "-".join(parts[1:-1])
    else:
        task = env_id
        data_quality = "unknown"
    return task, data_quality


def _apply_antmaze_defaults(config_dict):
    env_id = config_dict["env_id"]
    if "antmaze" not in env_id:
        return

    if config_dict["gamma"] == DEFAULT_GAMMA:
        config_dict["gamma"] = 0.999
    if config_dict["num_env_steps"] == DEFAULT_NUM_ENV_STEPS:
        config_dict["num_env_steps"] = 500001
    if config_dict["dep_targ"] == DEFAULT_DEP_TARG:
        config_dict["dep_targ"] = False
    if config_dict["normalise_state"] == DEFAULT_NORMALISE_STATE:
        config_dict["normalise_state"] = False

    if config_dict["offline"] and config_dict["eval_counter"] == DEFAULT_EVAL_COUNTER:
        config_dict["eval_counter"] = 10000

    config_dict["task"] = "ant"


def build_config(args, gym, np, torch):
    env = gym.make(args.env_id)
    task, data_quality = _parse_task_quality(args.env_id)
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    model_info = {
        "layers": [256, 256, 256],
        "hidden_activation": "ReLU",
        "critic_final_activation": "",
    }
    td3_params = {
        "policy_noise_std": 0.2,
        "noise_clip": 0.5,
        "policy_update_freq": 2,
        "exploration_noise_std": 0.1,
        "td3_alpha": 1,
    }

    config_dict = {
        "env": env,
        "hash_id": str(hash(time.time())),
        "seed": args.seed,
        "env_id": args.env_id,
        "gamma": args.gamma,
        "train_model": args.train_model,
        "model_info": model_info,
        "dep_targ": args.dep_targ,
        "ensemble_num": args.ensemble_num,
        "critic_factor": args.critic_factor,
        "n_processors": args.n_processors,
        "device": device,
        "num_evals": args.num_evals,
        "eval_counter": args.eval_counter,
        "num_env_steps": args.num_env_steps,
        "online_steps": args.online_steps,
        "optimiser": "Adam",
        "mem_size": args.mem_size,
        "batch_size": args.batch_size,
        "normalise_state": args.normalise_state,
        "gaussian_bc": args.gaussian_bc,
        "wandb_project": args.wandb_project,
        "id": str(uuid.uuid4())[:8],
        "task": task,
        "data_quality": data_quality,
        "offline": args.offline,
        "policy_stitch": args.policy_stitch,
        **td3_params,
    }

    _apply_antmaze_defaults(config_dict)
    config_dict["min_val"] = config_dict["env"].action_space.low
    config_dict["max_val"] = config_dict["env"].action_space.high

    seed = config_dict["seed"]
    config_dict["rng"] = np.random.default_rng(seed)
    config_dict["env"].action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    return config_dict


def _print_startup_summary(args, config_dict):
    mode = "offline" if config_dict["offline"] else "online"
    print(
        f"[run] algo={args.algo} env={args.env_id} mode={mode} "
        f"seed={args.seed} device={config_dict['device']}"
    )


def _print_config(config_dict):
    printable = {
        k: v
        for k, v in config_dict.items()
        if k not in {"env", "rng", "min_val", "max_val"}
    }
    pprint.pprint(printable, sort_dicts=True)


def main():
    args = parse_args()
    gym, np, torch, algo_runners, plot_functions = load_runtime_dependencies()
    ensure_d4rl_registered(args.env_id)
    config_dict = build_config(args, gym, np, torch)

    if args.print_config or args.dry_run:
        _print_config(config_dict)

    if args.dry_run:
        return

    _print_startup_summary(args, config_dict)
    runner = algo_runners[args.algo]
    try:
        runner(config_dict)
    finally:
        config_dict["env"].close()

    if args.plot == "return":
        plot_functions["return"](config_dict)
    elif args.plot == "std":
        plot_functions["std"](config_dict)


if __name__ == "__main__":
    main()
