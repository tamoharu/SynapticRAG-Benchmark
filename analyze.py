import json
import argparse

from config import parameters


param_ranges = {
    'cos_th': (0, 0.5),
    'v_th': (0, 0.2),
    'stimulus_th': (0, 0.2),
    'tau_scale': (0.1, 10),
}


def run(dataset, language):
    if dataset == 'SMRCs':
        from analyze.analyze_SMRCs import AnalyseProcess
    elif dataset == 'PerLTQA':
        from analyze.analyze_PerLTQA import AnalyseProcess
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    analyze = AnalyseProcess(language)
    
    params_to_analyze = [
        'cos_th',
        'v_th',
        'stimulus_th',
        'tau_scale',
    ]
    for param in params_to_analyze:
        param_step(analyze, parameters.sr_params, param)
    
    analyze.post()
    analyze.plot()


def param_step(analyze, params: dict, param: str):
    param_range = param_ranges[param]
    step = (param_range[1] - param_range[0]) / 10
    results = []
    for i in range(10):
        temp_params = params.copy()
        temp_params[param] = param_range[0] + step * i
        ratio = analyze.analyze(temp_params, param)
        result = {}
        result[param] = ratio
        results.append(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run evaluation process with given parameters.')
    parser.add_argument('--dataset', type=str, required=True, choices=['SMRCs', 'PerLTQA', 'QAConv'], help='Name of the dataset')
    parser.add_argument('--lang', type=str, required=False, choices=['en', 'ja', 'zh'], help='Language for the evaluation')
    args = parser.parse_args()
    run(args.dataset, args.lang)