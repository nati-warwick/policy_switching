# Policy Switching

run setup.sh to create a virtual environment and install necessary dependancies

- d4rl/gym compatibility patches are now applied automatically at runtime via `utils/compat.py`
- run `python smoke_test_compat.py --env-id halfcheetah-medium-v2` after installing dependencies to validate your local setup

- we also provide necessary mujoco files, move these to your home directory and rename directory `mujoco` to `.mujoco`

- we provide appendix of our work including a list of hyperparams

## Running code

To run code, execute `python main.py`

## Environment Notes

`d4rl` and `gym==0.26.x` can conflict during resolver time. Use:

1. `conda env create -f environment.yml`
2. `conda activate rlenv`
3. `pip install --no-deps git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl`

`setup.sh` automates this and also installs `mujoco-py`/`tqdm`, which are required for D4RL MuJoCo env registration.
It also installs `mjrl`, required by D4RL MuJoCo offline tasks such as `halfcheetah-medium-v2`.
On macOS, `setup.sh` forces Apple Clang + SDK flags for `mujoco-py` builds to avoid Homebrew `gcc` header errors.
`setup.sh` also pins `cython==0.29.36` during `mujoco-py` install because `mujoco-py==2.1.2.14` fails with Cython 3.x.
`setup.sh` force-reinstalls `numpy==1.26.4` after dependency setup to avoid Gym 0.26 incompatibility with NumPy 2.x (`np.bool8` removal).
Runtime startup also patches `mujoco-py` build flags on macOS to remove `-fopenmp` when using Apple Clang.
If you still see `command '/usr/local/bin/gcc-8' failed`, ensure no shell init script exports `CC`/`CXX` to gcc and rerun `./setup.sh`.


## Details
within `main.py` there is the option to run the following algos:
- `td3_n_offline` - Used to train td3_n offline
- `bc_offline` - Used to train bc agent (either gaussian or vanilla) offline
- `combined` - Used to combine policies for offline or online
- There are also a number of configurations that can be adjusted in `main.py` to control training

## Cite
--- 
Please cite our work if you find it useful

```
@inproceedings{neggatu2025evaluation,
  title={Evaluation-Time Policy Switching for Offline Reinforcement Learning},
  author={Neggatu, Natinael Solomon and Houssineau, Jeremie and Montana, Giovanni},
  booktitle={Proceedings of the 24th International Conference on Autonomous Agents and Multiagent Systems},
  pages={1520--1528},
  year={2025}
}

```
