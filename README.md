```text
                   _   _______ ______
    ____  __ ___  / | / / ___// ____/
   / __ \/ / __ \/  |/ /\__ \/ __/   
  / /_/ / / / / / /|  /___/ / /___   
 / .___/_/_/ /_/_/ |_//____/_____/   
/_/                                  
```

[![Build (master)](https://github.com/s9latimm/pinnse/actions/workflows/master.yml/badge.svg?branch=master)](https://github.com/s9latimm/pinnse/actions/workflows/master.yml)
[![Build (develop)](https://github.com/s9latimm/pinnse/actions/workflows/delevop.yml/badge.svg?branch=develop)](https://github.com/s9latimm/pinnse/actions/workflows/delevop.yml)

[Simulating Incompressible Flow with Physics-Informed Neural Networks](https://raw.githubusercontent.com/s9latimm/pinnse/master/paper_final.pdf)

## Codebase

[![UML](images/packages.svg)](https://github.com/s9latimm/pinnse/releases/latest/download/packages.pdf)

[![UML](images/classes.svg)](https://github.com/s9latimm/pinnse/releases/latest/download/classes.pdf)

## Setup

### Virtual Environment

#### Windows (Powershell)

```shell
$ python -m venv .venv
$ .\.activate.ps1
```


#### Linux

```shell
$ python -m venv .venv
$ source ./venv/bin/activate
```

### Dependencies

```shell
$ python -m pip install --upgrade pip
$ python -m pip install wheel
$ python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
$ python -m pip install -r requirements.txt
```

## Usage

- [Wikipedia](https://en.wikipedia.org/wiki/Navier%E2%80%93Stokes_equations#Incompressible_flow)

```text
usage: nse [-h] -E {empty,step,slalom,block,slit,cylinder,wing} [--inlet <u>] [--nu <nu>] [--rho <rho>] [--id <id>] [-N <train>] [-L <layers>] [-D {cpu,cuda}] [--supervised] [--dry] [-P] [-F] [-G] [-R]

options:
  -h, --help            show this help message and exit
  -L <layers>           size of layers seperated by colon (default: 100:100:100)

initialization:
  -E {empty,step,slalom,block,slit,cylinder,wing}
                        choose experiment
  --inlet <u>           set intake (default: 1.0)
  --nu <nu>             set viscosity (default: 0.1)
  --rho <rho>           set density (default: 1.0)

optimization:
  --id <id>             identifier / prefix for output directory (default: timestamp, example: 2024-10-28_09-26-09)
  -N <train>            number of optimization steps (default: 0)
  -D {cpu,cuda}         device used for training (default: cpu)
  --supervised          set training method to supervised approach (requires -F)
  --dry                 dry run

output:
  -P                    plot NSE in output directory
  -F                    initialize OpenFOAM experiment
  -G                    grade prediction (requires -F and -P)
  -R                    plot NSE with high resolution grid in output directory (requires -P)
```

#### Examples

```shell
$ python -m src.nse -E step --inlet 5 --nu .08 -N 100
```

```shell
$ python -m src.nse -E wing --id wing -L 100:100:100:100 --inlet 1 --nu .01 -D cuda -PRFGN 30000
```

```shell
$ python -m src.nse -E block
```

## References

- [Raissi, M. et al.: Physics Informed Deep Learning (Part II)](https://arxiv.org/pdf/1711.10566)
