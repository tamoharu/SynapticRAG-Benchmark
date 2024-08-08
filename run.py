import argparse
from config import parameters



def run(dataset, language):
    if dataset == 'SMRCs':
        from SMRCs.eval import EvalProcess
    elif dataset == 'PerLTQA':
        from PerLTQA.eval import EvalProcess
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    eval_process = EvalProcess(language)
    eval_process.eval_run(parameters.sr_params, parameters.ma_params, parameters.ma_ex_params, parameters.mb_ex_params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run evaluation process with given parameters.')
    parser.add_argument('--dataset', type=str, required=True, choices=['SMRCs', 'PerLTQA'], help='Name of the dataset')
    parser.add_argument('--lang', type=str, required=True, choices=['en', 'ja', 'zh'], help='Language for the evaluation')
    args = parser.parse_args()
    run(args.dataset, args.lang)