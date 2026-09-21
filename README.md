# Connect Four Search Agents

A CS MSc Introduction to Artificial Intelligence exercise implementing minimax, alpha-beta pruning, and expectimax agents for Connect Four. The supplied game engine, interface, and fixture cases make it possible to compare deterministic adversarial search with stochastic opponent modeling.

The repository includes the supplied game framework, utilities, graphics, and 14 test cases. My solution is in `src/multiAgents.py`.

The framework identifies John DeNero and Dan Klein as upstream contributors. The minimax, alpha-beta, and expectimax implementations in `src/multiAgents.py` are the submitted work.

## Implemented Agents

- `MinimaxAgent`: alternating maximum and minimum search
- `AlphaBetaAgent`: minimax with alpha-beta pruning
- `ExpectimaxAgent`: maximum search with uniform random opponent actions
- `BestRandom`: supplied baseline that selects among immediately scored moves

## Requirements

- Python 3.10 or newer
- [`uv`](https://docs.astral.sh/uv/)

## Setup

```bash
git clone https://github.com/KobieHazon/msc-intro-ai-connect-four.git
cd msc-intro-ai-connect-four
uv sync --dev
```

Run the fixture suite:

```bash
SDL_VIDEODRIVER=dummy uv run pytest
```

Run the original fixture driver directly:

```bash
SDL_VIDEODRIVER=dummy uv run python src/test.py
```

The dummy video driver avoids opening a window during automated tests. Interactive gameplay uses the supplied pygame interface.

## Repository layout

- `src/`: game framework, search-agent solution, and the fixture driver; original module names are retained.
- `tests/fixtures/`: all 14 fixture cases, unchanged.
- `tests/`: modern headless regression tests.

Run `SDL_VIDEODRIVER=dummy uv run python src/test.py` or the pytest command above from the repository root.
