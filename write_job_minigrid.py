import dicthash
import json
import numpy as np
import os
import sys
import argparse

sys.path.insert(0, '../../includes/')
import write_job_utils as utils


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
        'sim_script': 'minigrid.py',
        'dependencies': ['functions.py', 'network.py', 'operators.py'],

        # experiment configuration
        'prob_alteration_dict': {
            "alter_start_pos": 0,
            "alter_goal_pos": 0,
            "wall": 0.5,
            "lava": 0.5,
            "sand": 0.0,
        },

        # network parameterization
        'network_params': {
            'n_hidden': 30,
            'n_outputs': 3,  # Left, right, forward (pick up, drop, toggle, done are ingnored); env.action_space.n
            'learning_rate': 0.01,
            'weight_update_mode': 'evolved-rule',
            'beta': 1.0
        },

        # environment parameterization:
        'env_params': {
            'max_n_alterations': 4,
            'n_alterations_per_new_env': 3,
            'n_episodes_per_alteration': 2000,
            'seeds': np.linspace(1234567890, 1234567899, 4),
            'n_steps_max': 100,
            'temporal_novelty_decay': 0.99
        },

        # cgp parameterisation
        'max_time': 100, #82800,  # 82800s~23h
        'genome_params': {"n_inputs": 2, },
        'ea_params': {'n_processes': 4, },
    }

    params['md5_hash_sim_script'] = utils.md5_file(params['sim_script'])  # consistency check
    params['md5_hash_dependencies'] = [utils.md5_file(fn) for fn in params['dependencies']]  # consistency check

    results_folder = 'cgp_minigrid_run_with_cache'

    for use_rxet_init in [True, False]:

        params['use_rxet_init'] = use_rxet_init
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