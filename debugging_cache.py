import cgp


def debugging_objective(
        individual: cgp.IndividualSingleGenome,
        prob_alteration_dict: dict):
    individual.fitness = debugging_inner_objective(individual, prob_alteration_dict,)


@cgp.utils.disk_cache(
    "debugging_cache.pkl", compute_key=cgp.utils.compute_key_from_numpy_evaluation_and_args
)
def debugging_inner_objective(individual, prob_alteration_dict):
    sleep_time = 3.0


if __name__ == '__main__':

    prob_alteration_dict= {
        "alter_start_pos": 0,
        "alter_goal_pos": 0,
        "wall": 0.5,
        "lava": 0.5,
        "sand": 0.0,
    }