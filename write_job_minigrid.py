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
        'wall_clock_limit': '24:00:00',
        'ntasks': 6,
        'cpus-per-task': 4,
        'n_nodes': 1,
        'mail-user': 'henrik.mettler@unibe.ch',
        'account': 'ich029m',
        'constraint': 'mc',
        'partition': 'normal',
        'sim_script': 'minigrid_baseline.py',
        'dependencies': ['functions.py', 'network.py'],

        # experiment configuration
        'n_env_alterations': 6,
        'prob_alteration_dict': {
            "alter_start_pos": 0,
            "alter_goal_pos": 0,
            "wall": 0.5,
            "lava": 0.5,
            "sand": 0.0,
        }
    }

    params['md5_hash_sim_script'] = utils.md5_file(params['sim_script'])  # consistency check
    params['md5_hash_dependencies'] = [utils.md5_file(fn) for fn in params['dependencies']]  # consistency check

    results_folder = 'hidden_lr_scan'

    learning_rates = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]

    for learning_rate in learning_rates:
        params['learning_rate'] = learning_rate

        key = dicthash.generate_hash_from_dict(params)

        params['outputdir'] = os.path.join(os.getcwd(), results_folder, key)
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