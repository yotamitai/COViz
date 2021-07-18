import logging
from datetime import datetime
from os import makedirs
from os.path import abspath, exists, join


def get_logging(args):
    if not exists(abspath('logs')):
        makedirs('logs')
    file_name = '_'.join([datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(' ', '_'), args.name])
    log_name = join('logs', file_name)
    args.output = join('results', file_name)
    logging.basicConfig(filename=log_name + '.log', filemode='w',
                        format='%(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
    log(f'Comparing Agents: {args.name}', args.verbose)
    log(f'Disagreement importance by: {args.importance_type}', args.verbose)
    return args.name, file_name


def log(msg, verbose=False):
    if verbose: print(msg)
    logging.info(msg)