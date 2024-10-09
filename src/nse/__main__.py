import argparse
import logging
import sys

import cpuinfo
import psutil
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from src import OUTPUT_DIR, TIMESTAMP, ROOT_DIR, HIRES
from src.nse import DEFAULT_NU, DEFAULT_STEPS, DEFAULT_RHO, DEFAULT_INTAKE
from src.nse.experiments import EXPERIMENTS
from src.nse.experiments.experiment import NSEExperiment
from src.nse.model import NSEModel
from src.nse.plot import plot_foam, plot_prediction, plot_history, plot_geometry
from src.utils.timer import Stopwatch


def main(
    experiment: NSEExperiment,
    n: int,
    plot: bool,
    identifier: str,
    device: str,
    foam: bool,
    hires: bool,
    save: bool,
    layers: list[int],
) -> None:
    logging.info(f'NU:         {experiment.nu:.3E}')
    logging.info(f'RHO:        {experiment.rho:.3E}')
    logging.info(f'INTAKE:     {experiment.inlet:.3E}')
    logging.info(f'GRID:       {experiment.knowledge.mesh().shape}')
    logging.info(f'DIMENSIONS: {experiment.dim}')
    logging.info(f'HIRES:      {HIRES}')
    logging.info(f'STEPS:      {args.train}')
    logging.info(f'TIMESTAMP:  {TIMESTAMP}')
    logging.info(f'OUTPUT:     {(OUTPUT_DIR / identifier).relative_to(ROOT_DIR)}')

    logging.info(f'CPU:        {cpuinfo.get_cpu_info()["brand_raw"]} ')
    logging.info(f'LOGICAL:    {psutil.cpu_count(logical=True)}')
    logging.info(f'MEMORY:     {psutil.virtual_memory().total / (1024 ** 3):.2f} GB')

    if foam:
        logging.info('PLOT: OPENFOAM')
        plot_foam(experiment, identifier)

    if plot:
        logging.info('PLOT: GEOMETRY')
        plot_geometry(experiment, identifier)

    model = NSEModel(experiment, device, n, layers)

    logging.info(model)
    logging.info(f'PARAMETERS: {len(model)}')

    if n > 0:
        with Stopwatch(lambda t: logging.info(f'TIME: {t}')):
            with tqdm(total=n, position=0, leave=True) as pbar, logging_redirect_tqdm():

                def callback(history):
                    if pbar.n < n:
                        if pbar.n > 0:
                            if pbar.n % 1e2 == 0:
                                change = history[-1].mean() - history[-2].mean()
                                logging.info(f'  {pbar.n:{len(str(n))}d}: {history[-1].mean():18.16f} {change:+.3E}')
                            if plot and pbar.n % 1e3 == 0:
                                logging.info('PLOT: PREDICTION')
                                plot_prediction(pbar.n, experiment, model, identifier)
                                logging.info('PLOT: LOSS')
                                plot_history(pbar.n, experiment, model, identifier)
                            if hires and pbar.n % 1e4 == 0:
                                logging.info('PLOT: HIRES PREDICTION')
                                plot_prediction(pbar.n, experiment, model, identifier, hires=True)
                        pbar.update(1)

                logging.info(f'TRAINING: START {n}')
                model.train(callback)
                logging.info(f'TRAINING: END {pbar.n}')

        if save:
            model.save(OUTPUT_DIR / identifier / 'model.pt')

    model.eval()

    if plot:
        logging.info('PLOT: PREDICTION')
        plot_prediction(n, experiment, model, identifier)
        logging.info('PLOT: LOSS')
        plot_history(pbar.n, experiment, model, identifier)

    if hires:
        logging.info('PLOT: HIRES PREDICTION')
        plot_prediction(n, experiment, model, identifier, hires=True)

    if experiment.supervised:
        logging.info(f'NU: {model.nu:.16f}')
        logging.info(f'RHO: {model.rho:.16f}')

    # if foam:
    #     logging.info('PLOT: DIFFERENCE')
    #     plot_diff(n, data, model, identifier)


def parse_cmd() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog='nse')

    initialization = parser.add_argument_group('initialization')
    initialization.add_argument(
        '-e',
        '--experiment',
        type=str,
        choices=EXPERIMENTS.keys(),
        required=True,
        help='choose experiment',
    )
    initialization.add_argument(
        '-i',
        '--intake',
        type=float,
        metavar='<intake>',
        default=DEFAULT_INTAKE,
        help='set intake [m/s]',
    )
    initialization.add_argument(
        '--nu',
        type=float,
        metavar='<nu>',
        default=DEFAULT_NU,
        help='set viscosity [m^2/s]',
    )
    initialization.add_argument(
        '--rho',
        type=float,
        metavar='<rho>',
        default=DEFAULT_RHO,
        help='set density [kg/m^2]',
    )

    optimization = parser.add_argument_group('optimization')
    optimization.add_argument(
        '--id',
        type=str,
        metavar='<id>',
        default=TIMESTAMP,
        help='identifier / prefix for output directory',
    )
    optimization.add_argument(
        '-n',
        '--train',
        type=int,
        metavar='<train>',
        default=DEFAULT_STEPS,
        help='number of optimization steps',
    )

    def layers_type(arg: str) -> list[int]:
        try:
            layers = [int(i.strip()) for i in arg.split(':') if len(i.strip()) > 0]
            if len(layers) < 1 or min(layers) < 1:
                parser.error(f"argument -l/--layers: invalid value: '{arg}'")
            return layers
        except (TypeError, ValueError):
            parser.error(f"argument -l/--layers: invalid value: '{arg}'")

    parser.add_argument(
        '-l',
        '--layers',
        type=layers_type,
        metavar='<layers>',
        default='100:100:100',
        help='size of layers seperated by colon (default: 100:100:100)',
    )
    optimization.add_argument(
        '-d',
        '--device',
        type=str,
        choices=['cpu', 'cuda'],
        default='cpu',
        help='device used for training (default: cpu)',
    )
    optimization.add_argument(
        '-f',
        '--foam',
        action='store_true',
        default=False,
        help='load OpenFOAM',
    )
    optimization.add_argument(
        '--supervised',
        action='store_true',
        default=False,
        help='set training method to supervised approach (requires OpenFOAM)',
    )

    output = parser.add_argument_group('output')
    output.add_argument(
        '-p',
        '--plot',
        action='store_true',
        default=False,
        help='plot NSE in output directory',
    )
    output.add_argument(
        '-r',
        '--hires',
        action='store_true',
        default=False,
        help='plot NSE with high resolution grid in output directory',
    )
    output.add_argument(
        '--save',
        action='store_true',
        default=False,
        help='store model parameters in output directory',
    )

    args = parser.parse_args()

    if args.hires and not args.plot:
        parser.error('the following arguments are required: --plot')

    if args.supervised and not args.foam:
        parser.error('the following arguments are required: --foam')

    return args


if __name__ == '__main__':
    args = parse_cmd()
    if args.plot:
        (OUTPUT_DIR / args.id).mkdir(parents=True, exist_ok=args.id != TIMESTAMP)
        logging.basicConfig(format='%(message)s',
                            handlers=[
                                logging.FileHandler(OUTPUT_DIR / args.id / 'log.txt', mode='w'),
                                logging.StreamHandler(sys.stdout)
                            ],
                            encoding='utf-8',
                            level=logging.INFO)
    else:
        logging.basicConfig(format='%(message)s',
                            handlers=[logging.StreamHandler(sys.stdout)],
                            encoding='utf-8',
                            level=logging.INFO)

    try:
        main(
            EXPERIMENTS[args.experiment](
                args.nu,
                args.rho,
                args.intake,
                args.foam,
                args.supervised,
            ),
            args.train,
            args.plot,
            args.id,
            args.device,
            args.foam,
            args.hires,
            args.save,
            args.layers,
        )
        logging.info('EXIT: SUCCESS')
    except KeyboardInterrupt:
        logging.info('EXIT: ABORT')
