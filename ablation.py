import argparse

from config import parameters


def run(dataset, language):
    if dataset == 'SMRCs':
        from ablation.ablation_SMRCs import AblationProcess
    elif dataset == 'PerLTQA':
        from ablation.ablation_PerLTQA import AblationProcess
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    ablation = AblationProcess(language)
    ablation.ablation(parameters.sr_params_stim, True, False, False)
    ablation.ablation(parameters.sr_params_lif, False, True, False)
    ablation.ablation(parameters.sr_params_prop, False, False, True)
    ablation.post()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run evaluation process with given parameters.')
    parser.add_argument('--dataset', type=str, required=True, choices=['SMRCs', 'PerLTQA', 'QAConv'], help='Name of the dataset')
    parser.add_argument('--lang', type=str, required=False, choices=['en', 'ja', 'zh'], help='Language for the evaluation')
    args = parser.parse_args()
    run(args.dataset, args.lang)