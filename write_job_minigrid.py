import dicthash
import numpy as np
import os
import sys

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

        # network parameterization
        'network_params': {
            'n_hidden': 400,
            'n_outputs': 3,  # Left, right, forward (pick up, drop, toggle, done are ingnored); env.action_space.n
            'learning_rate_inp2hid': 0.0,
            'learning_rate_hid2out': 0.01,
            'weight_update_mode': 'evolved-rule',
            'beta': 1.0
        },

        # curriculum parameterization:
        'curriculum_params': {
            'max_n_alterations': 4,
            'n_alterations_per_new_env': 3,
            'n_episodes_per_alteration': 2000,
            'n_steps_max': 100,
            'temporal_novelty_decay': 0.99,
            'spatial_novelty_time_decay': 0.99,
            'spatial_novelty_distance_decay': 0.5,
            'prob_alteration_dict': {
                "alter_start_pos": 0,
                "alter_goal_pos": 0,
                "wall": 0.5,
                "lava": 0.5,
                "sand": 0.0,
            },
        },
        # seed parameters
        # 'seeds': np.linspace(1234567890, 1234567899, 4),

        # cgp parameterisation
        'max_time':  40000,  # 82800s~23h
        'genome_params': {"n_inputs": 5, },
        'ea_params': {'n_processes': 4, },
        #'use_rxet_init': True,

    }

    params['md5_hash_sim_script'] = utils.md5_file(params['sim_script'])  # consistency check
    params['md5_hash_dependencies'] = [utils.md5_file(fn) for fn in params['dependencies']]  # consistency check

    results_folder = 'optimized_run_time_12h'

    initial_seed_array = [1234567810, 1234567820, 1234567830, 1234567840, 1234567850, 1234567860, 1234567870, 1234567880, 1234567890]

    for use_drxeot_init in [True]:

        params['use_drxeot_init'] = use_drxeot_init

        for initial_seed in initial_seed_array:

            params['seeds'] = np.linspace(initial_seed, initial_seed+3, 4)

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
