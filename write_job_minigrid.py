import dicthash
import json
import numpy as np
import os
import sys
import argparse

sys.path.insert(0, '../../includes/')
import write_job_utils as utils

"""parser = argparse.ArgumentParser
parser.add_argument("--n_episodes", required=True)
parser.add_argument("--n_hidden", required=True)
parser.add_argument("--learning_rate", required=True)
args = parser.parse_args()"""

if __name__ == '__main__':

    params = {

        # machine setup
        'submit_command': 'sbatch',
        'jobfile_template': 'jobfile_template.jdf',
        'jobname': 'rrl',
        'wall_clock_limit': '00:30:00',
        'ntasks': 6,
        'cpus-per-task': 4,
        'n_nodes': 1,
        'mail-user': 'henrik.mettler@unibe.ch',
        'account': 'ich029m',
        'constraint': 'mc',
        'partition': 'debug',
        'sim_script': 'minigrid_baseline.py',
        'dependencies': ['functions.py', 'network.py'],

        # experiment configuration
        'seed' :  123456789,
        'n_hidden' : sys.argv[0],
        'learning_rate' : sys.argv[1],
    }

    params['md5_hash_sim_script'] = utils.md5_file(params['sim_script'])  # consistency check
    params['md5_hash_dependencies'] = [utils.md5_file(fn) for fn in params['dependencies']]  # consistency check

    key = dicthash.generate_hash_from_dict(params)

    params['outputdir'] = os.path.join(os.getcwd(), key)
    params['workingdir'] = os.getcwd()

    submit_job = True

    print('preparing job')
    print(' ', params['outputdir'])

    utils.mkdirp(params['outputdir'])
    utils.write_pickle(params, os.path.join(params['outputdir'], 'params.pickle'))
    utils.create_jobfile(params)
    utils.copy_file(params['sim_script'], params['outputdir'])
    utils.copy_files(params['dependencies'], params['outputdir'])
    if submit_job:
        print('submitting job')
        utils.submit_job(params)