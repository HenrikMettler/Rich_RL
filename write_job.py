import dicthash
import json
import numpy as np
import os
import sys

import cgp

sys.path.insert(0, '../../includes/')
import write_job_utils as utils

from operators import Const05Node, Const2Node

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
        'sim_script': 'main.py',
        'dependencies': ['functions.py', 'operators.py', 'network.py'],

        # experiment configuration
        'seed' : 12345,
        'gamma' : 0.9,
        'use_online_init' : True,
        'n_episodes' : 1000,
        'cum_reward_threshold' : 250000,  # empirical value that policy gradient receives after ~ 3000 episodes
        'n_episodes_reward_expectation' : 100,

        'population_params' : {"n_parents": 1, "seed": 12345},

        'ea_params' : {"n_offsprings": 4, "mutation_rate": 0.03, "reorder_genome": True, "n_processes": 1,
                 "hurdle_percentile": [0.5, 0.0], },
        'evolve_params' : {"max_generations": 2},  # Todo: set reasonable termination fitness

        'genome_params' : {
            "n_inputs": 4,  # reward, el_traces, done (episode termination), expected_cum_reward_episode
            "n_outputs": 1,
            "n_columns": 300,
            "n_rows": 1,
            "levels_back": None,
            "primitives": (cgp.Mul, cgp.Add, cgp.Sub, cgp.ConstantFloat, Const05Node, Const2Node)
            }

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
