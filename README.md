# Harry Potter — Logical Inference (Gringotts)

This project is a Python implementation of logical inference in the Harry Potter world.  
Harry attempts to find a Deathly Hallow hidden in Gringotts vaults while avoiding dragons and traps.

## Features
- Partial observability: Harry only discovers the map while moving
- Observations: vaults, dragons, and sulfur smell near traps
- Actions: move, destroy trap, wait, collect
- Inference-based strategy to safely reach the correct vault
- Time-constrained controller for each action

## Project Structure
- ex2.py — your implementation (GringottsController with inference logic)
- check.py — checker that runs the controller
- utils.py — helper functions
- inputs.py — example input cases

## Requirements
- Python 3.10+
- No external libraries (standard library only)

## How to Run
```bash
python check.py
