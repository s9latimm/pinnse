import argparse
import logging
import sys

import cpuinfo
import psutil
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import src.navier_stokes.config as config
from src.navier_stokes.geometry import NavierStokesGeometry
from src.navier_stokes.model import NavierStokesModel
from src.navier_stokes.plot import plot_foam, plot_prediction, plot_diff, plot_hires, plot_history, plot_geometry
from src.utils.timer import Stopwatch


def main(n: int, plot: bool, identifier: str, device: str, foam: bool, hires: bool, intake: float):
    data = NavierStokesGeometry(args.nu, args.rho, intake)

    logging.info(f'NU:        {data.nu:.3E}')
    logging.info(f'RHO:       {data.rho:.3E}')
    logging.info(f'INTAKE:    {intake:.3E}')
    logging.info(f'GRID:      {config.GRID}')
    logging.info(f'GEOMETRY:  {config.GEOMETRY}')
    logging.info(f'HIRES:     {config.HIRES}')
    logging.info(f'STEPS:     {args.train}')
    logging.info(f'TIMESTAMP: {config.TIMESTAMP}')
    logging.info(f'OUTPUT:    {(config.OUTPUT_DIR / identifier).relative_to(config.ROOT_DIR)}')

    logging.info(f'CPU:       {cpuinfo.get_cpu_info()["brand_raw"]} ')
    logging.info(f'LOGICAL:   {psutil.cpu_count(logical=True)}')
    logging.info(f'MEMORY:    {psutil.virtual_memory().total / (1024 ** 3):.2f} GB')

    if foam:
        plot_foam(data, identifier)

    if plot:
        plot_geometry(data, identifier)

    model = NavierStokesModel(data, device, n)

    logging.info(model)

    if n > 0:
        with Stopwatch(lambda t: logging.info(f'TIME: {t}')):
            with tqdm(total=n, position=0, leave=True) as pbar, logging_redirect_tqdm():

                def callback(history):
                    if pbar.n < n:
                        if pbar.n > 0:
                            if pbar.n % 1e2 == 0:
                                change = history[-1].mean() - history[-2].mean()
                                logging.info(f'  {pbar.n:{len(str(n))}d}: {history[-1].mean():20.16f} {change:+20.16f}')
                            if plot and pbar.n % 1e3 == 0:
                                plot_prediction(pbar.n, data, model, identifier)
                                plot_history(pbar.n, data, model, identifier)
                        pbar.update(1)

                logging.info(f'TRAINING: START {n}')
                model.train(callback)
                logging.info(f'TRAINING: END {pbar.n}')

        if plot:
            model.save(config.OUTPUT_DIR / identifier / 'model.pt')

    model.eval()

    if plot:
        logging.info('PLOT: PREDICTION')
        plot_prediction(n, data, model, identifier)
        plot_history(pbar.n, data, model, identifier)

    if hires:
        logging.info('PLOT: HIRES PREDICTION')
        plot_hires(n, data, model, identifier)

    if foam:
        logging.info('PLOT: DIFFERENCE')
        plot_diff(n, data, model, identifier)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='main')
    parser.add_argument(
        '-n',
        '--train',
        type=int,
        metavar='<train>',
        default=config.DEFAULT_STEPS,
    )
    parser.add_argument(
        '--device',
        type=str,
        metavar='<device>',
        default='cpu',
    )
    parser.add_argument(
        '--nu',
        type=float,
        metavar='<nu>',
        default=config.DEFAULT_NU,
    )
    parser.add_argument(
        '--rho',
        type=float,
        metavar='<rho>',
        default=config.DEFAULT_RHO,
    )
    parser.add_argument(
        '--id',
        type=str,
        metavar='<id>',
        default=config.TIMESTAMP,
    )
    parser.add_argument(
        '-i',
        '--intake',
        type=float,
        metavar='<intake>',
        default=1.,
    )
    parser.add_argument(
        '-p',
        '--plot',
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '-r',
        '--hires',
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '-f',
        '--foam',
        action='store_true',
        default=False,
    )
    args = parser.parse_args()

    if args.plot:
        (config.OUTPUT_DIR / args.id).mkdir(parents=True, exist_ok=args.id != config.TIMESTAMP)
        logging.basicConfig(format='%(message)s',
                            handlers=[
                                logging.FileHandler(config.OUTPUT_DIR / args.id / 'log.txt', mode='w'),
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
        main(args.train, args.plot, args.id, args.device, args.foam, args.hires, args.intake)
        logging.info('EXIT: SUCCESS')
    except KeyboardInterrupt:
        logging.info('EXIT: ABORT')
