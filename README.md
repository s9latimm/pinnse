# Inf MP AOS

## References

- [Raissi, M. et al.: Physics Informed Deep Learning (Part II)](https://arxiv.org/pdf/1711.10566)

## Requirements

### Virtual Environment

#### Windows

```shell
$ python -m venv .venv
$ .\.activate.ps1
```


#### Linux

```shell
$ python -m venv .venv
$ source ./venv/bin/activate
```

### Packages

```shell
$ python -m pip install --upgrade pip
$ python -m pip install wheel
$ python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
$ python -m pip install -r requirements.txt
```

## Navier-Stokes

- [Incompressible Flow](https://en.wikipedia.org/wiki/Navier%E2%80%93Stokes_equations#Incompressible_flow)

```
usage: nse [-h] [-i <intake>] [--nu <nu>] [--rho <rho>] [--id <id>] [-n <train>] [-d {cpu,cuda}] [-f] [--supervised] [-p] [-r] [--save]

options:
  -h, --help            show this help message and exit

initialization:
  -i <intake>, --intake <intake>
                        set intake [m/s]
  --nu <nu>             set viscosity [m^2/s]
  --rho <rho>           set density [kg/m^2]

optimization:
  --id <id>             identifier / prefix for output directory
  -n <train>, --train <train>
                        number of optimization steps
  -d {cpu,cuda}, --device {cpu,cuda}
                        device used for training
  -f, --foam            load OpenFOAM
  --supervised          set training method to supervised approach (requires OpenFOAM)

output:
  -p, --plot            plot NSE in output directory
  -r, --hires           plot NSE with high resolution grid in output directory
  --save                store model parameters in output directory
```

### Examples

```shell
$ python -m src.nse --id eval -i 1.2 --nu 1 -pn 10000 -d cuda
```

```shell
$ python -m src.nse -i 10 --nu .068 -prfn 5000 --supervised
```

```shell
$ python -m src.nse --id test
```
