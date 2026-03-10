"""CLI entry point for the GLUE mislabeled-data experiments."""

import argh, os, pickle
from simulator import main
from configs import *

parser = argh.ArghParser()

def run(exp_id='', run_id=0, runpath=''):
    """Load a preset config, persist it, and execute the selected run."""
    _, runs=eval(f'config_{exp_id}()')
    config=runs[run_id]
    if runpath != '':
        config['runpath']=runpath 
        os.chdir(runpath)  # Resolve relative outputs from the requested working directory.
    with open('config.pickle', 'wb') as pkl_file:
        pickle.dump(config, pkl_file)  # Snapshot the exact run config for later inspection.
    main(config)

parser = argh.ArghParser()
parser.add_commands([run])

if __name__ == '__main__':
    parser.dispatch()

