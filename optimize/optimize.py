import sys
sys.path.append('..')
import os
import optuna
import argparse
from tqdm import tqdm
from config import parameters

from eval.eval_SMRCs import EvalProcess


opt_log_path = os.path.abspath(os.path.join(os.path.dirname(__file__), './opt_log/optimize_log.txt'))

sample_count = 10
n_jobs = -1
n_trials = 200


def progress_bar_callback(study, trial):
    pbar.update(1)


def sr_objective(trial, eval_process):
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

    score = eval_process.sr_opt_run(trial.number, sample_count, params)
    print(f'Trial: {trial.number}, Score: {score}, Params: {params}')
    return score


def ma_objective(trial, eval_process):
    cos_th = trial.suggest_float('cos_th', 0.0, 1.0)

    params = {
        'cos_th': cos_th,
    }
    
    score = eval_process.ma_opt_run(trial.number, sample_count, params)
    print(f'Trial: {trial.number}, Score: {score}')
    return score


def ma_ex_objective(trial, eval_process):
    cos_th = trial.suggest_float('cos_th', 0.0, 1.0)
    r_scale = trial.suggest_float('r_scale', -10, 20)
    t_scale = trial.suggest_float('t_scale', -10, 20)
    g_scale = trial.suggest_float('g_scale', -10, 20)

    params = {
        'cos_th': cos_th,
        'r_scale': r_scale,
        't_scale': t_scale,
        'g_scale': g_scale
    }
    
    score = eval_process.ma_ex_opt_run(trial.number, sample_count, params)
    print(f'Trial: {trial.number}, Score: {score}')
    return score


def mb_ex_objective(trial, eval_process):
    top_k = trial.suggest_int('top_k', 1, 10)
    forget_th = trial.suggest_float('forget_th', 0.0, 1.0)
    t_scale = trial.suggest_float('t_scale', -10, 20)
    s_scale = trial.suggest_float('s_scale', -10, 20)
    s_init = trial.suggest_float('s_init', 0, 20)

    params = {
        'top_k': top_k,
        'forget_th': forget_th,
        't_scale': t_scale,
        's_scale': s_scale,
        's_init': s_init
    }

    score = eval_process.mb_ex_opt_run(trial.number, sample_count, params)
    print(f'Trial: {trial.number}, Score: {score}')
    return score


def run(model):
    eval_process = EvalProcess('en')
    eval_process.clear_directory()

    objectives = {
        'sr': sr_objective,
        'ma': ma_objective,
        'ma_ex': ma_ex_objective,
        'mb_ex': mb_ex_objective
    }

    if model not in objectives:
        raise ValueError(f"Unsupported model: {model}")

    global pbar
    pbar = tqdm(total=n_trials)

    objective_with_eval = lambda trial: objectives[model](trial, eval_process)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_with_eval, n_trials=n_trials, n_jobs=n_jobs, callbacks=[progress_bar_callback])
    pbar.close()
    eval_process.clear_directory()

    best_trials = [trial for trial in study.trials if trial.value == study.best_value]

    best_eval_score = -float('inf')
    best_eval_params = None

    for trial in best_trials:
        print(trial.params)

    with open(opt_log_path, 'a') as f:
        f.write(f'Best trial: {best_eval_params}\n')
        f.write(f'Best value: {best_eval_score}\n\n')
    print(f'Best trial: {best_eval_params}')
    print(f'Best value: {best_eval_score}')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run evaluation process with given parameters.')
    parser.add_argument('--model', type=str, required=True, choices=['sr', 'ma', 'ma_ex', 'mb_ex'], help='Model for the optimization')
    args = parser.parse_args()
    run(args.model)
