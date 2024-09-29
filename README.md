# Inf MP AOS

## References

- [Raissi, M. et al.: Physics Informed Deep Learning (Part II)](https://arxiv.org/pdf/1711.10566)

## Requirements

```shell
$ python -m pip install --upgrade pip
$ python -m pip install wheel
$ python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
$ python -m pip install -r requirements.txt
```

## Navier-Stokes

- [Incompressible Flow](https://en.wikipedia.org/wiki/Navier%E2%80%93Stokes_equations#Incompressible_flow)

```
usage: main [-h] [-n <train>] [--device <device>] [--nu <nu>] [--rho <rho>] [--id <id>] [-i <intake>] [-p] [-r] [--save] [-f]

options:
  -h, --help            show this help message and exit
  -n <train>, --train <train>
  --device <device>
  --nu <nu>
  --rho <rho>
  --id <id>
  -i <intake>, --intake <intake>
  -p, --plot
  -r, --hires
  --save
  -f, --foam
```

### Example

This command runs a session called `test` with 10 training steps using an intake of 1 and viscosity of 0.04.
In addition, it plots a high-resolution prediction and compares the output to OpenFOAM.

```shell
$ python -m src.nse --id eval -i 1.2 --nu .02 -prfn 10000 --device cuda
```

