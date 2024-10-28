import argparse
import logging
import sys

import cpuinfo
import psutil
import torch
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from src import OUTPUT_DIR, TIMESTAMP, ROOT_DIR, HIRES
from src.nse import DEFAULT_NU, DEFAULT_STEPS, DEFAULT_RHO, DEFAULT_U
from src.nse.controller.simulation import Simulation
from src.nse.model.experiments import EXPERIMENTS
from src.nse.model.experiments.experiment import Experiment
from src.nse.view.grading import plot_setup, plot_foam, grade, export
from src.nse.view.prediction import plot_prediction, plot_losses
from src.utils.timer import Stopwatch


def main(
    experiment: Experiment,
    n: int,
    plot: bool,
    identifier: str,
    device: str,
    foam: bool,
    hires: bool,
    layers: list[int],
    grading: bool,
) -> None:
    if experiment.supervised:
        logging.warning('running supervised experiment')
    if foam:
        logging.info(f'FOAM:       {experiment.foam.name}')
    logging.info(f'NU:         {experiment.nu:.3E}')
    logging.info(f'RHO:        {experiment.rho:.3E}')
    logging.info(f'INLET:      {experiment.inlet_f}')
    logging.info(f'DIMENSION:  {experiment.shape}')
    logging.info(f'MESH:       {experiment.knowledge.grid().shape}')
    logging.info(f'HIRES:      {HIRES}')
    logging.info(f'STEPS:      {n}')
    logging.info(f'TIMESTAMP:  {TIMESTAMP}')
    logging.info(f'OUTPUT:     {(OUTPUT_DIR / identifier).relative_to(ROOT_DIR)}')
    logging.info(f'EXPERIMENT: {experiment.name}')

    suffix = f'{device}__{layers[0]}__{len(layers)}__{n}'
    if grading:
        logging.info(f'SUFFIX:     {suffix}')

    if torch.cpu.is_available():
        info = cpuinfo.get_cpu_info()
        logging.info(f'CPU:        {info["brand_raw"]}')
        logging.info(f'LOGICAL:    {info["count"]}')
        l2 = -(info['l2_cache_line_size'] * 1000) // -1024
        logging.info(f'L2:         {(-(info["l2_cache_size"]) // -1024) // l2} x {l2} KB')
        logging.info(f'L3:         {-(info["l3_cache_size"] // -(1024 ** 2))} MB')
        logging.info(f'RAM:        {-(psutil.virtual_memory().total // -(1024 ** 3))} GB')

    if torch.cuda.is_available():
        info = torch.cuda.get_device_properties(torch.cuda.device(0))
        logging.info(f'GPU:        {info.name}')
        logging.info(f'CUDA:       {info.multi_processor_count}')
        logging.info(f'MEMORY:     {-(info.total_memory // -(1024 ** 3))} GB')

    model = Simulation(experiment, device, n, layers)

    logging.info(model)
    logging.info(f'PARAMETERS: {len(model)}')

    logging.info('PLOT: Setup')
    plot_setup(experiment, identifier)

    timer = Stopwatch()
    if n > 0:
        with Stopwatch(lambda t: logging.info(f'TIME: {t}')) as timer:
            with tqdm(total=n, position=0, leave=True) as pbar, logging_redirect_tqdm():

                def callback(loss: list[list[float]]) -> None:
                    if pbar.n < n:
                        if pbar.n > 0:
                            if pbar.n % 1e2 == 0:
                                change = loss[-1][0] - loss[-2][0]
                                logging.info(f'  {pbar.n:{len(str(n))}d}: {loss[-1][0]:18.16f} {change:+.3E}')
                            if plot and pbar.n % 1e4 == 0:
                                logging.info('PLOT: Prediction')
                                plot_prediction(pbar.n, experiment, model, identifier)
                                logging.info('PLOT: Loss')
                                plot_losses(pbar.n, model, identifier)
                            # if hires and pbar.n % 1e4 == 0:
                            #     logging.info('PLOT: HiRes Prediction')
                            #     plot_prediction(pbar.n, experiment, model, identifier, hires=True)
                        pbar.update(1)

                logging.info(f'TRAINING: Start {n}')
                model.train(callback)
                logging.info(f'TRAINING: End {pbar.n}')

        model.eval()

        if plot:
            logging.info('PLOT: Prediction')
            plot_prediction(n, experiment, model, identifier)
            logging.info('PLOT: Loss')
            plot_losses(pbar.n, model, identifier)

            if hires:
                logging.info('PLOT: HiRes Prediction')
                plot_prediction(n, experiment, model, identifier, hires=True)

        if experiment.supervised:
            logging.info(f'NU: {model.nu:.16f}')
            logging.info(f'RHO: {model.rho:.16f}')

        if grading:
            logging.info('EXPORT')
            export(timer, experiment, model, identifier, suffix)

    if foam:
        logging.info('PLOT: openFOAM')
        plot_foam(experiment, identifier)

    if grading:
        logging.info('GRADING')
        grade(experiment, identifier, suffix)


def parse_cmd() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog='nse')

    initialization = parser.add_argument_group('initialization')
    initialization.add_argument(
        '-E',
        type=str,
        choices=EXPERIMENTS.keys(),
        required=True,
        help='choose experiment',
    )
    initialization.add_argument(
        '--inlet',
        type=float,
        metavar='<u>',
        default=DEFAULT_U,
        help=f'set intake (default: {DEFAULT_U})',
    )
    initialization.add_argument(
        '--nu',
        type=float,
        metavar='<nu>',
        default=DEFAULT_NU,
        help=f'set viscosity (default: {DEFAULT_NU})',
    )
    initialization.add_argument(
        '--rho',
        type=float,
        metavar='<rho>',
        default=DEFAULT_RHO,
        help=f'set density (default: {DEFAULT_RHO})',
    )

    optimization = parser.add_argument_group('optimization')
    optimization.add_argument(
        '--id',
        type=str,
        metavar='<id>',
        default=TIMESTAMP,
        help=f'identifier / prefix for output directory (default: timestamp, example: {TIMESTAMP})',
    )
    optimization.add_argument(
        '-N',
        type=int,
        metavar='<train>',
        default=DEFAULT_STEPS,
        help=f'number of optimization steps (default: {DEFAULT_STEPS})',
    )

    def layers_type(arg: str) -> list[int]:
        try:
            layers = [int(i.strip()) for i in arg.split(':') if len(i.strip()) > 0]
            if len(layers) < 1 or min(layers) < 1:
                parser.error(f'argument -l/--layers: invalid value: \'{arg}\'')
            return layers
        except (TypeError, ValueError):
            parser.error(f'argument -l/--layers: invalid value: \'{arg}\'')
        return []

    parser.add_argument(
        '-L',
        type=layers_type,
        metavar='<layers>',
        default='100:100:100',
        help='size of layers seperated by colon (default: 100:100:100)',
    )
    optimization.add_argument(
        '-D',
        type=str,
        choices=['cpu', 'cuda'],
        default='cpu',
        help='device used for training (default: cpu)',
    )
    optimization.add_argument(
        '--supervised',
        action='store_true',
        default=False,
        help='set training method to supervised approach (requires -F)',
    )
    optimization.add_argument(
        '--dry',
        action='store_true',
        default=False,
        help='dry run',
    )

    output = parser.add_argument_group('output')
    output.add_argument(
        '-P',
        action='store_true',
        default=False,
        help='plot NSE in output directory',
    )
    output.add_argument(
        '-F',
        action='store_true',
        default=False,
        help='initialize OpenFOAM experiment',
    )
    output.add_argument(
        '-G',
        action='store_true',
        default=False,
        help='grade prediction (requires -F and -P)',
    )
    output.add_argument(
        '-R',
        action='store_true',
        default=False,
        help='plot NSE with high resolution grid in output directory (requires -P)',
    )

    args = parser.parse_args()

    if args.N < 0:
        parser.error(f'argument -N: invalid value: \'{args.train}\'')

    if (args.R or args.G) and not args.P:
        parser.error('the following arguments are required: -P')

    if (args.supervised or args.G) and not args.F:
        parser.error('the following arguments are required: -F')

    return args


if __name__ == '__main__':
    cmd = parse_cmd()
    handler = cmd.id.split('/')[-1]
    if cmd.P:
        (OUTPUT_DIR / cmd.id).mkdir(parents=True, exist_ok=cmd.id != TIMESTAMP)
        logging.basicConfig(format=f'{handler}: %(message)s',
                            handlers=[
                                logging.FileHandler(OUTPUT_DIR / cmd.id / 'log.txt', mode='w'),
                                logging.StreamHandler(sys.stdout)
                            ],
                            encoding='utf-8',
                            level=logging.INFO)
    else:
        logging.basicConfig(format=f'{handler}: %(message)s',
                            handlers=[logging.StreamHandler(sys.stdout)],
                            encoding='utf-8',
                            level=logging.INFO)
    # logging.info(f'CMD: {cmd}')

    try:
        main(
            EXPERIMENTS[cmd.E](
                cmd.nu,
                cmd.rho,
                cmd.inlet,
                cmd.supervised,
            ),
            1 if cmd.dry else cmd.N,
            cmd.P,
            cmd.id,
            cmd.D,
            cmd.F,
            cmd.R,
            cmd.L,
            cmd.G,
        )
        logging.info('EXIT: Success')
    except KeyboardInterrupt:
        logging.info('EXIT: Abort')
