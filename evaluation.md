## System

FOAM:

- TODO

pinNSE:

- CPU:     AMD Ryzen 9 5950X 16-Core Processor
- LOGICAL: 32
- L2:      16 x 500 KB
- L3:      64 MB
- RAM:     64 GB
- GPU:     NVIDIA GeForce RTX 3080
- CUDA:    68
- MEMORY:  10 GB


## Precision

- select example from FOAM dataset
- try to reproduce in pinNSE
- vary mesh and inlet / nu factor

| experiment | inlet | nu  |         FOAM         |      pinNSE 3x100       | pinNSE 4x80 |
|:-----------|-------|-----|:--------------------:|:-----------------------:|-------------|
| step       | 1     | .01 | convergence (21 min) | convergence (6 s, += 3) |             |
|            | 4     | .04 |                      |                         |             |
|            |       |     |                      |                         |             |
|            |       |     |                      |                         |             |
|            |       |     |                      |                         |             |
|            |       |     |                      |                         |             |
|            |       |     |                      |                         |             |
|            |       |     |                      |                         |             |


## Point-Of-Failure

- fixed grid
- decrease viscosity until no longer convergence or able to predict

| experiment | inlet | nu  |            FOAM             |            pinNSE 3x100            | pinNSE 4x80 |
|:-----------|-------|-----|:---------------------------:|:----------------------------------:|-------------|
| empty      | 1     | .01 | convergence (100 it, 9 min) | convergence (200 it, 10 min, += 4) |             |
|            | 1     | .02 |         x (20 min)          |                                    |             |
|            | 1     |     |                             |                                    |             |
|            |       |     |                             |                                    |             |
|            |       |     |                             |                                    |             |
|            |       |     |                             |                                    |             |
|            |       |     |                             |                                    |             |
|            |       |     |                             |                                    |             |
