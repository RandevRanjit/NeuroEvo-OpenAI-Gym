# NeuroEvo-OpenAI-Gym

Neuroevolution written with NumPy alone and tried on three OpenAI Gym environments. The networks are trained only by selection and mutation, without gradients.
A population of small feed-forward networks plays OpenAI Gym episodes, each network's score over its episode is its fitness, and the next generation keeps the best networks and adds mutated copies of the top one's weights and biases.

Status: 2021 project, kept for reference.

## Environments

There are three runner/agent pairs.
Each `main_game_runner*.py` imports the agent file with the same suffix.

| Runner | Agent | Environment | Layer widths | Population | Runs? |
|---|---|---|---|---|---|
| `main_game_runner.py` | `neural_network_NE_agent.py` | `LunarLander-v2` | 8, 16, 8, 4 | 40 | Yes |
| `main_game_runner_2.py` | `neural_network_NE_agent_2.py` | `BipedalWalkerHardcore-v3` | 24, 64, 64, 64, 64, 64, 4, plus a zero-input layer | 10 | No, crashes on the first step |
| `main_game_runner_3.py` | `neural_network_NE_agent_3.py` | Procgen Chaser (`procgen:procgen-chaser-v0`) | 12288, 64, 56, 49, 42, 36, 31, 27, 23, 20, 17, 15 | 10 | Yes |

LunarLander and Chaser take the argmax of the output as a discrete action (4 and 15 choices), while the BipedalWalker version passes the softmax vector straight through as its 4 continuous actions.
Chaser sees raw pixels: the 64×64×3 RGB frame, flattened, values 0 to 255.

## How the algorithm works

The networks are plain NumPy dense layers.
Weights start as `0.1 * randn` and biases as `randn`.
Every layer applies ReLU, the output layer included, and a softmax over the last layer gives the output.
After the first hidden layer, v1 halves the width at each layer and v3 takes 7/8 of it (floored), stopping once the next width would be at or below the output size; the output layer comes after that.

Each generation:

1. Every network plays one episode. Its fitness is the reward summed over that episode (but see the off-by-one under Known issues).
2. Networks are ranked by fitness.
3. The next population is half children (rounded up), then unchanged copies of the top-ranked networks, then fresh random networks. With 40 genomes (v1) that comes to 20 children, 18 copies and 2 new random networks; with 10 genomes (v2, v3) it is 5 children and 5 copies.
4. The top network's weights and biases are meant to go to `weights_best.txt` and `biases_best.txt` in the working directory. Each file is emptied through one file handle but written through another, buffered one opened at import, so `weights_best.txt` usually holds the previous generation's best and `biases_best.txt` is usually empty or holds many generations' dumps run together.

`mutatations()` builds each child one gene at a time, and weights and biases go through the same steps.
These are the starting values:

| Variable | Value | What it controls |
|---|---|---|
| `randomization_rate` | 0.1978 | chance the gene is replaced with `uniform(-1, 1)` |
| `breed_rate` | 0.3 | otherwise, chance of the "parent 2" branch over the "parent 1" branch |
| `randomization_rate_for_parent_2` | 0.6 | chance of noise in the parent 2 branch, half-width m/2 |
| `randomization_rate_for_parent_1` | 0.09 | chance of noise in the parent 1 branch, half-width m/8 |
| `number_of_children_per_fam` | 2 | children per parent pairing |

Here m is the mean of the top network's weights (or biases for a bias gene).
It is a signed mean, so it sits close to zero.
For a freshly initialised v1 network the m/2 weight half-width is typically around 0.002, small next to the `uniform(-1, 1)` resets that supply most of the variation.
At these settings about 20% of genes are reset, 14% get the wider noise, 5% get the narrower noise and the other 61% are copied unchanged.

The loops are written for two-parent crossover, but every child is built from the top-ranked network (see Known issues), so in practice this is elitist, mutation-only evolution.

Mutation strength also adapts.
`create_generation()` tracks the best score seen so far.
In v2 and v3 a stagnation counter goes up each generation that fails to beat it, and restarts at 1 on a generation that does.
Once the counter passes 2 it goes back to 0 and the "environmental stress" level rises by one: the parent 2 noise chance goes up by 0.015, the parent 1 noise chance by 0.055 and `breed_rate` drops by 0.015, each only while its offset is inside a hard-coded limit, and children per family goes up by 1 with no cap.
Every new record takes one level off again.
v1 only bumps the counter when a generation's best exactly equals the record, so with continuous rewards its stress rule fires only on an exact tie.

## How the files relate

All the code went up in a single upload on 8 December 2021, so git has no history of how it changed.
The numbered files are successive iterations, kept as copies:

- `neural_network_NE_agent.py` with `main_game_runner.py` is the first version: LunarLander, 40 genomes, halving layer widths.
- The `_2` pair moves to BipedalWalker Hardcore. The population drops to 10, the layers become a fixed 64 wide, the output becomes continuous, and the stagnation rule changes to the one described above.
- The `_3` pair is the latest iteration: Procgen Chaser from pixels, back to tapering layers (the 7/8 ratio) and an argmax over 15 actions.

The selection and mutation code (`sort_weights_and_biases` and `mutatations`) is the same in all three agent files.
v1 and v3 are the two that run end to end.
The comment block at the top of each agent file is an older plan for a keyboard-driven chaser game with 38 inputs and W/A/S/D outputs; it doesn't describe any of the three Gym set-ups.

## Running it

The code needs the gym API from before 0.26, where `reset()` returns only the observation and `step()` returns four values.
`main_game_runner.py` also calls `envs.registry.all()`, which is gone in gym 0.26: it fails on line 4 with `AttributeError: 'dict' object has no attribute 'all'`.
The versions below use gym 0.21.0, the current release when this code was uploaded, on Python 3.9.

```bash
python3.9 -m venv .venv
source .venv/bin/activate
# gym 0.21.0 only installs with pip < 24.1 and an older setuptools
pip install "pip<24.1" "setuptools==65.5.0" "wheel==0.38.4"
pip install "gym==0.21.0" "Box2D==2.3.10" "numpy<1.24" matplotlib
```

`Box2D==2.3.10` ships prebuilt wheels.
gym's own `box2d` extra asks for `box2d-py==2.3.5` instead, which has to be compiled with SWIG on any recent Python.

LunarLander:

```bash
pip install "pyglet==1.5.27"
python main_game_runner.py
```

This runner calls `env.render()` on every step, so it opens a window and needs pyglet.
gym 0.21's renderer uses legacy OpenGL calls (`glBegin`, `glPushMatrix`) that pyglet 2 dropped, hence the 1.5 pin.

Procgen Chaser:

```bash
pip install "procgen==0.10.7"
python main_game_runner_3.py
```

procgen 0.10.7 only has x86-64 wheels (Linux, Intel macOS and Windows) for Python 3.7 to 3.10, so on Apple Silicon, install it into an x86-64 Python running under Rosetta.
The runner imports `Box2D` and `gym3` without using them, so both have to be installed.
Its `env.render()` calls do nothing: procgen opens a window only when the environment is created with `render_mode="human"`, and this runner doesn't pass it.

`main_game_runner_2.py` stops on its first step (see below).

Each generation prints the sorted `(score, network)` list, the population size and the generation number.
Nothing reads `weights_best.txt` back in, so a run can't be resumed.
For the larger networks NumPy also abbreviates the arrays with `...`, which means the dump isn't a full copy of the weights.

In a 90-second headless run on an Apple Silicon Mac, the LunarLander version got through about 500 generations.
The best single episode scored 165 and every generation's mean score stayed negative.
gym's reward threshold for `LunarLander-v2` is 200, so this doesn't solve the task.

## Known issues

These are left as they were in 2021.

- `neural_network_NE_agent_2.py` builds a final layer with zero inputs, because the loop that set its width is commented out. The first forward pass fails with `ValueError: shapes (1,4) and (0,4) not aligned`.
- No crossover happens. Both branches in `mutatations()` read the gene from `parent_b`, and the outer parent loop exits after its first pass, where parent A and parent B are both the top-ranked network. Every child is a mutated copy of the best genome.
- Rewards are credited one step late. The runner passes the reward from the previous `step()` in before each action, so the reward from an episode's last step (LunarLander's -100 for crashing or +100 for coming to rest) is added to the next network's score.
- Raising children per family changes nothing, because the number of children is capped at half the population.
- The best-genome dumps are unreliable. Both files are opened once at import in append mode and truncated each generation through a separate `open(..., "w")`, and the buffered writes land later, so the files lag behind or pile up several generations.
- Importing an agent module also runs `evolution_controller(100)` at the bottom of the file. It builds a population and discards it; the plotting loop inside is commented out.

## Licence

MIT, see `LICENSE`.
