import os
import json
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from config import parameters
from utils.eval import Eval


class Analyze(Eval):
    def __init__(self):
        super().__init__()
        self.analyze_dir = ''
        self.result_dir = ''

    def post(self):
        param_dirs = os.listdir(self.analyze_dir)
        all_results = {}
        for param_dir in param_dirs:
            if os.path.isfile(self.analyze_dir + '/' + param_dir):
                continue
            param_files = os.listdir(self.analyze_dir + '/' + param_dir)
            param_results = {}
            for param_file in param_files:
                param_value = param_file.replace('.json', '')
                path = self.analyze_dir + '/' + param_dir + '/' + param_file
                param_results[param_value] = self.calc_score(path)
            all_results[param_dir] = param_results
        with open(self.result_dir + '/analyze.json', 'w') as f:
            json.dump(all_results, f, indent=4)

    def calc_score(self, path):
        task_count = 0
        sr_count = 0
        rag_count = 0
        results_def = self.read_json_data(self.result_dir + '/memories.json')
        sr_search = self.read_json_data(path)
        for result in results_def:
            task_count += len(result['recall'])
            index = result['index']
            sr = {}
            for search in sr_search:
                if search['index'] == index:
                    sr = search['sr']
                    break
            sr_results = {}
            for sr_result in sr.items():
                if sr_result[1] >= 1000:
                    sr_results[sr_result[0]] = sr_result[1]
            for sr_result in sr_results:
                if int(sr_result) in result['recall']:
                    sr_count += 1

            rag_results = self.get_top_k_indices(len(sr_results), result['rag'])
            for rag_result in rag_results:
                if int(rag_result) in result['recall']:
                    rag_count += 1
        sr_count /= task_count
        rag_count /= task_count
        return sr_count - rag_count

    def plot(self):
        data_dict = self.read_analyze_files()
        params = list(list(data_dict.values())[0].keys())
        datasets = list(data_dict.keys())
        colors = ['#FF9F00', '#00A9FF', '#00BE8C', '#FF69B4']
        markers = ['o', 's', '^', 'D']
        highlight_values = {
            'cos_th': parameters.sr_params['cos_th'],
            'v_th': parameters.sr_params['v_th'],
            'stimulus_th': parameters.sr_params['stimulus_th'],
            'tau_scale': parameters.sr_params['tau_scale'],
        }
        
        plt.rcParams['font.size'] = 16
        plt.rcParams['axes.labelsize'] = 18
        plt.rcParams['legend.fontsize'] = 16
        fig = plt.figure(figsize=(12, 10))
    
        lines = []
        labels = []
        for param_idx, param in enumerate(params):
            ax = plt.subplot(2, 2, param_idx + 1)
            for ds_idx, dataset in enumerate(datasets):
                param_data = data_dict[dataset][param]
                x_values = [float(x) for x in param_data.keys()]
                y_values = list(param_data.values())
                sorted_indices = np.argsort(x_values)
                x_sorted = np.array(x_values)[sorted_indices]
                y_sorted = np.array(y_values)[sorted_indices]
                
                line = ax.plot(x_sorted, y_sorted, '-', 
                    color=colors[ds_idx], 
                    marker=markers[ds_idx],
                    markersize=6,
                    linewidth=2,
                    alpha=0.9)[0]
                
                if param_idx == 0:
                    lines.append(line)
                    labels.append(dataset)
                        
            if param in highlight_values:
                y_min = min([min(data_dict[ds][param].values()) for ds in datasets])
                y_max = max([max(data_dict[ds][param].values()) for ds in datasets])
                x_min = min([min(float(x) for x in data_dict[ds][param].keys()) for ds in datasets])
                x_max = max([max(float(x) for x in data_dict[ds][param].keys()) for ds in datasets])
    
                ax.axvline(x=highlight_values[param], color='black', linestyle='--', linewidth=2)
                if param == 'v_th':
                    ax.text(highlight_values[param] + (x_max - x_min) * 0.03, y_max - (y_max - y_min) * 0.5, 
                        'x=0.099', 
                        va='bottom',
                        ha='left',
                        color='black',
                        fontsize=16)
                elif param == 'tau_scale':
                    ax.text(highlight_values[param] + (x_max - x_min) * 0.03, y_max - (y_max - y_min) * 0.8, 
                        f'x={highlight_values[param]:.3f}', 
                        va='bottom',
                        ha='left',
                        color='black',
                        fontsize=16)
                elif param == 'cos_th':
                    ax.text(highlight_values[param] + (x_max - x_min) * -0.03, y_max - (y_max - y_min) * 0.6, 
                        f'x={highlight_values[param]:.3f}', 
                        va='bottom',
                        ha='right',
                        color='black',
                        fontsize=16)                    
                else:
                    ax.text(highlight_values[param] + (x_max - x_min) * 0.03, y_max - (y_max - y_min) * 0.7,  
                        f'x={highlight_values[param]:.3f}', 
                        va='bottom',
                        ha='left',
                        color='black',
                        fontsize=16)
                    
                for spine in ax.spines.values():
                    spine.set_linewidth(2)
                    
            param_names = {
                'cos_th': 'cos_th (Cosine Similarity Threshold)',
                'v_th': 'v_th (Firing Threshold)',
                'stimulus_th': 'stim_th (Stimulus Threshold)',
                'tau_scale': 'tau_scale (Time Constant Scale)'
            }
            ax.set_xlabel(param_names[param], labelpad=10)
            ax.set_ylabel('Δ%', labelpad=10)
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.tick_params(axis='x', which='both', length=0, pad=8)
            ax.tick_params(axis='y', which='both', length=0, pad=8) 
            
            x_min = min([min(float(x) for x in data_dict[ds][param].keys()) for ds in datasets])
            x_max = max([max(float(x) for x in data_dict[ds][param].keys()) for ds in datasets])
            x_margin = (x_max - x_min) * 0.05
            ax.set_xlim(x_min - x_margin, x_max + x_margin)
            y_min = min([min(data_dict[ds][param].values()) for ds in datasets])
            y_max = max([max(data_dict[ds][param].values()) for ds in datasets])
            y_margin = (y_max - y_min) * 0.05
            ax.set_ylim(y_min - y_margin, y_max + y_margin)
            
        optimal_line = mlines.Line2D([], [], color='black', linestyle='--', linewidth=2, label='Default Setting')
        lines.append(optimal_line)
        labels.append('Default Setting')
            
        fig.legend(lines, labels, 
                loc='upper center', 
                bbox_to_anchor=(0.5, 0.975),
                ncol=5,
                columnspacing=0.8,
                frameon=False)
        
        plt.tight_layout(pad=2.0)
        plt.subplots_adjust(top=0.9, hspace=0.3)
        
        save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../results/parameter_sensitivity.png'))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return True

    def read_analyze_files(self):
        result_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../results/'))
        results = {}
        for dataset_folder in os.listdir(result_path):
            if os.path.isfile(result_path + '/' + dataset_folder):
                continue
            for file in os.listdir(result_path + '/' + dataset_folder):
                if file == 'analyze.json':
                    with open(result_path + '/' + dataset_folder + '/' + file) as f:
                        results[dataset_folder] = json.load(f)
        return results