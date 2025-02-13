import sys
sys.path.append('..')
import os
import optuna
import argparse
from tqdm import tqdm

from ablation.ablation_SMRCs import AblationProcess


opt_log_path = os.path.abspath(os.path.join(os.path.dirname(__file__), './opt_log/optimize_log.txt'))

sample_count = 10
n_jobs = -1
n_trials =200


def progress_bar_callback(study, trial):
    pbar.update(1)


def sr_objective(trial, eval_process, ab_stim, ab_lif, ab_prop):
    cos_th = trial.suggest_float('cos_th', 0, 0.5)
    v_th = trial.suggest_float('v_th', 0.0, 0.2)
    stimulus_th = trial.suggest_float('stimulus_th', 0.0, 0.2)
    tau_init = trial.suggest_float('tau_init', 40, 50)
    tau_scale = trial.suggest_float('tau_scale', 0.0, 10.0)
    time_scale = trial.suggest_float('time_scale', 0.0, 10.0)
    bond_scale = trial.suggest_float('bond_scale', 0.0, 10)
    v_rest = trial.suggest_float('v_rest', -1, 10.0)
    i_rest = trial.suggest_float('i_rest', -10, 1)
    params = {
        'cos_th': cos_th,
        'v_th': v_th,
        'stimulus_th': stimulus_th,
        'tau_init': tau_init,
        'tau_scale': tau_scale,
        'time_scale': time_scale,
        'bond_scale': bond_scale,
        'v_rest': v_rest,
        'i_rest': i_rest
    }

    score = eval_process.sr_opt_run(trial.number, sample_count, params, ab_stim, ab_lif, ab_prop)
    print(f'Trial: {trial.number}, Score: {score}, Params: {params}')
    return score


def run(model):
    eval_process = AblationProcess('en')
    eval_process.clear_directory()

    global pbar
    pbar = tqdm(total=n_trials)

    ab_stim = False
    ab_lif = False
    ab_prop = False

    if model == 'stim':
        ab_stim = True
    elif model == 'lif':
        ab_lif = True
    elif model == 'prop':
        ab_prop = True

    objective_with_eval = lambda trial: sr_objective(trial, eval_process, ab_stim=ab_stim, ab_lif=ab_lif, ab_prop=ab_prop)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_with_eval, n_trials=n_trials, n_jobs=n_jobs, callbacks=[progress_bar_callback])
    pbar.close()
    eval_process.clear_directory()

    best_trials = [trial for trial in study.trials if trial.value == study.best_value]

    best_eval_score = -float('inf')

    for trial in best_trials:
        print(trial.params)

    with open(opt_log_path, 'a') as f:
        f.write(f'Best value: {best_eval_score}\n\n')
    print(f'Best value: {best_eval_score}')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run evaluation process with given parameters.')
    parser.add_argument('--model', type=str, required=True, choices=['stim', 'lif', 'prop'], help='Model for the optimization')
    args = parser.parse_args()
    run(args.model)
