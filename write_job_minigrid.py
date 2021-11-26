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
        }
    }


    params['md5_hash_sim_script'] = utils.md5_file(params['sim_script'])  # consistency check
    params['md5_hash_dependencies'] = [utils.md5_file(fn) for fn in params['dependencies']]  # consistency check

    results_folder = 'cgp_minigrid_run'
    #results_folder = 'comp_autograd_eq4'

    #weight_update_modes = ['equation4', 'autograd']
    for use_rxet_init in [False, True]:

    #for weight_update_mode in weight_update_modes:
        for reset in [True, False]:

            params['weight_update_mode'] = weight_update_mode
            params['network_reset_after_alteration'] = reset
            #params['use_rxet_init'] = use_rxet_init
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