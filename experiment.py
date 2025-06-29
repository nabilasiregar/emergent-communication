import copy
import itertools
import argparse
import yaml
import game
import pdb

SEEDS = [123, 2025]

EXPERIMENTS = {
    "bee_default": [{}],
    "human_vocab_sweep": [
        {"vocab_size": 20},
        {"vocab_size": 50},
        {"vocab_size": 100},
    ],
    "human_maxlen_sweep": [
        {"max_len": 2},
        {"max_len": 4},
        {"max_len": 6},
        {"max_len": 10},
    ],
    "human_default": [{}],
}

def load_base_config(path):
    print(f"Loading base config from: '{path}'")
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def config_to_cli_params(config):
    params = []
    for key, value in config.items():
        if isinstance(value, bool) and value:
            params.append(f'--{key}')
        elif not isinstance(value, bool):
            params.extend([f'--{key}', str(value)])
    return params

def main():
    parser = argparse.ArgumentParser(description="Experiment Script")
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        choices=EXPERIMENTS.keys(),
        help="The name of the experiment set to run."
    )
    args = parser.parse_args()

    # determine which config file to use based on the experiment name
    if args.name.startswith('human'):
        base_config = load_base_config('config_human.yml')
    elif args.name.startswith('bee'):
        base_config = load_base_config('config_bee.yml')
    else:
        raise ValueError(f"Experiment name '{args.name}' must start with 'bee_' or 'human_'")

    experiment_set = EXPERIMENTS[args.name]
    print(f"Starting Experiment Set: '{args.name}'")

    run_counter = 1
    total_runs = len(experiment_set) * len(SEEDS)

    for param_variation in experiment_set:
        for seed in SEEDS:
            print("\n" + "="*50)
            print(f"Running ({run_counter}/{total_runs}): Seed={seed}, Params={param_variation}")
            print("="*50)

            current_config = copy.deepcopy(base_config)
            current_config.update(param_variation)
            current_config['random_seed'] = seed
            current_config['final_run'] = True
            
            # create a descriptive name for logging
            name_parts = [args.name]
            for key, val in param_variation.items():
                name_parts.append(f"{key}{val}")
            name_parts.append(f"seed{seed}")
            
            experiment_name = "_".join(name_parts)

            cli_params = config_to_cli_params(current_config)

            game.main(cli_params, experiment_name)
            
            run_counter += 1

    print(f"\n--- Experiment '{args.name}' Finished! ---")

if __name__ == "__main__":
    main()
