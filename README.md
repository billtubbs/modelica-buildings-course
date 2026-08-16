# modelica-buildings-course

Code and materials for the Modelica-based simulation of building and district energy systems course, held at Aalborg University, 26–28 August 2026.

Course page: https://phd.moodle.aau.dk/blocks/vitrina/detail.php?id=2974

## Installation

This repo uses a conda environment (`environment.yaml`) to install Python plus the CBC solver, and `pyproject.toml` to declare the Python package dependencies.

1. Create the environment:
   ```
   conda env create -f environment.yaml
   ```

2. Activate it:
   ```
   conda activate mod-build
   ```

3. Install this repo's dependencies:
   ```
   pip install -e .
   ```

## Dependencies

- Python 3.11
- [CBC solver](https://github.com/coin-or/Cbc) (`coincbc`, via conda-forge)
- [oemof.solph](https://oemof-solph.readthedocs.io/)
- [Modelon Impact Client](https://modelon-impact-client.readthedocs.io/en/latest/index.html)
- ipython, jupyter, numpy, matplotlib, pandas, scipy, casadi, pyyaml
- pytest (for running tests)
