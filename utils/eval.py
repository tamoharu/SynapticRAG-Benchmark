import os
import shutil
import json
import numpy
import random
random.seed(42)
import pandas as pd
import matplotlib.pyplot as plt

from config import words
from utils.log import Log
from utils.memory_model import MemoryModel
from utils.timer import average_timer


log_level = 'error'


class Eval:
    def __init__(self):
        self.log = Log(log_level=log_level)
        self.memory = MemoryModel()
        self.conv_dir = ''
        self.result_dir = ''

    def result_log(self, path, section, trigger_index, recall_indices, sr_results, rag_results, ma_results, ma_ex_results, mb_results, mb_ex_results):
        model_indices = list(sr_results.keys())
        with open(path, 'a') as f:
            f.write(f'section: {section}\n')
            f.write(f'trigger: {trigger_index}\n')
            f.write(f'recall: {recall_indices}\n')
            f.write(f'SynapticRAG_results: {model_indices}\n')
            f.write(f'RAG_results: {rag_results}\n')
            f.write(f'MyAgent_results: {ma_results}\n')
            f.write(f'MyAgent_ex_results: {ma_ex_results}\n')
            f.write(f'MemoryBank_results: {mb_results}\n')
            f.write(f'MemoryBank_ex_results: {mb_ex_results}\n\n')
    
    def erc_post(self):
        with open(self.result_dir + '/memories.json', 'r') as f:
            results = json.load(f)
        task_count = 0

        sr = 0
        ma = 0
        ma_ex = 0
        mb = 0
        mb_ex = 0
        rag = 0

        ma_mg = 0
        ma_ex_mg = 0
        mb_mg = 0
        mb_ex_mg = 0
        rag_mg = 0

        for result in results:
            task_count += len(result['recall'])

            sr_results = {}
            for sr_result in result['sr'].items():
                if sr_result[1] >= 1000:
                    sr_results[sr_result[0]] = sr_result[1]
            for sr_result in sr_results:
                if int(sr_result) in result['recall']:
                    sr += 1
            
            ma_results = self.get_top_k_indices(len(sr_results), result['ma'])
            for ma_result in ma_results:
                if int(ma_result) in result['recall']:
                    ma += 1

            ma_ex_results = self.get_top_k_indices(len(sr_results), result['ma_ex'])
            for ma_ex_result in ma_ex_results:
                if int(ma_ex_result) in result['recall']:
                    ma_ex += 1
            
            mb_results = self.get_top_k_indices(len(sr_results), result['mb'])
            for mb_result in mb_results:
                if int(mb_result) in result['recall']:
                    mb += 1

            mb_ex_results = self.get_top_k_indices(len(sr_results), result['mb_ex'])
            for mb_ex_result in mb_ex_results:
                if int(mb_ex_result) in result['recall']:
                    mb_ex += 1
            
            rag_results = self.get_top_k_indices(len(sr_results), result['rag'])
            for rag_result in rag_results:
                if int(rag_result) in result['recall']:
                    rag += 1

            ma_results_mg = self.get_top_k_indices(max(len(sr_results), len(result['recall'])), result['ma'])
            for ma_result in ma_results_mg:
                if int(ma_result) in result['recall']:
                    ma_mg += 1
            
            ma_ex_results_mg = self.get_top_k_indices(max(len(sr_results), len(result['recall'])), result['ma_ex'])
            for ma_ex_result in ma_ex_results_mg:
                if int(ma_ex_result) in result['recall']:
                    ma_ex_mg += 1

            mb_results_mg = self.get_top_k_indices(max(len(sr_results), len(result['recall'])), result['mb'])
            for mb_result in mb_results_mg:
                if int(mb_result) in result['recall']:
                    mb_mg += 1

            mb_ex_results_mg = self.get_top_k_indices(max(len(sr_results), len(result['recall'])), result['mb_ex'])
            for mb_ex_result in mb_ex_results_mg:
                if int(mb_ex_result) in result['recall']:
                    mb_ex_mg += 1

            rag_results_mg = self.get_top_k_indices(max(len(sr_results), len(result['recall'])), result['rag'])
            for rag_result in rag_results_mg:
                if int(rag_result) in result['recall']:
                    rag_mg += 1
        
        sr /= task_count
        ma /= task_count
        ma_ex /= task_count
        mb /= task_count
        mb_ex /= task_count
        rag /= task_count

        ma_mg /= task_count
        ma_ex_mg /= task_count
        mb_mg /= task_count
        mb_ex_mg /= task_count
        rag_mg /= task_count

        with open(self.result_dir + '/erc_result.json', 'w') as f:
            json.dump({
                'sr': sr,
                'ma': ma,
                'ma_ex': ma_ex,
                'mb': mb,
                'mb_ex': mb_ex,
                'rag': rag,
                'ma_mg': ma_mg,
                'ma_ex_mg': ma_ex_mg,
                'mb_mg': mb_mg,
                'mb_ex_mg': mb_ex_mg,
                'rag_mg': rag_mg
            }, f, indent=4)
        

    def k_post(self):
        with open(self.result_dir + '/memories.json', 'r') as f:
            results = json.load(f)
        min_k = 0
        max_k = 0

        recall_sr = 0
        recall_ma = 0
        recall_ma_ex = 0
        recall_mb = 0
        recall_mb_ex = 0
        recall_rag = 0

        precision_sr = 0
        precision_ma = 0
        precision_ma_ex = 0
        precision_mb = 0
        precision_mb_ex = 0
        precision_rag = 0

        for result in results:
            min_k = max(min_k, len(result['recall']))
            max_k = max(max_k, len(result['sr']))

        k_results = []
        for k in range(min_k, max_k + 1):
            for result in results:
                tp_sr = 0
                sr_results = self.get_top_k_indices(k, result['sr'])
                for sr_result in sr_results:
                    if int(sr_result) in result['recall']:
                        tp_sr += 1
                recall_sr += tp_sr / len(result['recall'])
                precision_sr += tp_sr / k

                tp_ma = 0
                ma_results = self.get_top_k_indices(k, result['ma'])
                for ma_result in ma_results:
                    if int(ma_result) in result['recall']:
                        tp_ma += 1
                recall_ma += tp_ma / len(result['recall'])
                precision_ma += tp_ma / k

                tp_ma_ex = 0
                ma_ex_results = self.get_top_k_indices(k, result['ma_ex'])
                for ma_ex_result in ma_ex_results:
                    if int(ma_ex_result) in result['recall']:
                        tp_ma_ex += 1
                recall_ma_ex += tp_ma_ex / len(result['recall'])
                precision_ma_ex += tp_ma_ex / k

                tp_mb = 0
                mb_results = self.get_top_k_indices(k, result['mb'])
                for mb_result in mb_results:
                    if int(mb_result) in result['recall']:
                        tp_mb += 1
                recall_mb += tp_mb / len(result['recall'])
                precision_mb += tp_mb / k

                tp_mb_ex = 0
                mb_ex_results = self.get_top_k_indices(k, result['mb_ex'])
                for mb_ex_result in mb_ex_results:
                    if int(mb_ex_result) in result['recall']:
                        tp_mb_ex += 1
                recall_mb_ex += tp_mb_ex / len(result['recall'])
                precision_mb_ex += tp_mb_ex / k

                tp_rag = 0
                rag_results = self.get_top_k_indices(k, result['rag'])
                for rag_result in rag_results:
                    if int(rag_result) in result['recall']:
                        tp_rag += 1
                recall_rag += tp_rag / len(result['recall'])
                precision_rag += tp_rag / k

            recall_sr /= len(results)
            recall_ma /= len(results)
            recall_ma_ex /= len(results)
            recall_mb /= len(results)
            recall_mb_ex /= len(results)
            recall_rag /= len(results)

            precision_sr /= len(results)
            precision_ma /= len(results)
            precision_ma_ex /= len(results)
            precision_mb /= len(results)
            precision_mb_ex /= len(results)
            precision_rag /= len(results)

            if recall_sr >= 0.99 or recall_ma >= 0.99 or recall_ma_ex >= 0.99 or recall_mb >= 0.99 or recall_mb_ex >= 0.99 or recall_rag >= 0.99:
                break

            k_results.append({
                'k': k,
                'recall_SynapticRAG': recall_sr,
                'recall_RAG': recall_rag,
                'recall_MyAgent': recall_ma,
                'recall_MyAgent(ext)': recall_ma_ex,
                'recall_MemoryBank': recall_mb,
                'recall_MemoryBank(ext)': recall_mb_ex,
                'precision_SynapticRAG': precision_sr,
                'precision_RAG': precision_rag,
                'precision_MyAgent': precision_ma,
                'precision_MyAgent(ext)': precision_ma_ex,
                'precision_MemoryBank': precision_mb,
                'precision_MemoryBank(ext)': precision_mb_ex
            })

        with open(self.result_dir + '/k_result.json', 'w') as f:
            json.dump(k_results, f, indent=4)


    def _k_post(self):
        # memories.jsonには、"results"というキーの下にdictが格納されたリストがあると仮定する
        with open(self.result_dir + '/memories.json', 'r') as f:
            data = json.load(f)
        # もしファイル直下がリストの場合は、以下の行をコメントアウトしてください
        results = data  

        # 正解ラベルの数（recallリストの長さ）ごとにグループ分け
        groups = {}
        for result in results:
            recall_len = len(result['recall'])
            groups.setdefault(recall_len, []).append(result)

        # 各グループのタスク数を表示
        for recall_len, group in sorted(groups.items()):
            print(f"recall数が{recall_len}のタスク数: {len(group)}")

        # グループごとに、kの値ごとの評価結果を保存する辞書を用意
        k_results_by_group = {}

        # グループ（recallの長さ）ごとに処理
        for recall_len, group in groups.items():
            group_results = []
            # グループ内の各resultにおいて、kの最小値は正解ラベルの数（＝recall_len）、
            # 最大値は各resultの'sr'リストの長さの最大値とする
            group_min_k = recall_len
            group_max_k = max(len(result['sr']) for result in group)
            
            # kの値を変化させながら評価
            for k in range(group_min_k, group_max_k + 1):
                # 各手法ごとの合計値を初期化
                recall_sr = recall_ma = recall_ma_ex = recall_mb = recall_mb_ex = recall_rag = 0
                precision_sr = precision_ma = precision_ma_ex = precision_mb = precision_mb_ex = precision_rag = 0

                for result in group:
                    # 各resultについて、各手法の上位k件を取得し、正解との一致数を数える
                    tp_sr = 0
                    sr_results = self.get_top_k_indices(k, result['sr'])
                    for sr_result in sr_results:
                        if int(sr_result) in result['recall']:
                            tp_sr += 1
                    recall_sr += tp_sr / len(result['recall'])
                    precision_sr += tp_sr / k

                    tp_ma = 0
                    ma_results = self.get_top_k_indices(k, result['ma'])
                    for ma_result in ma_results:
                        if int(ma_result) in result['recall']:
                            tp_ma += 1
                    recall_ma += tp_ma / len(result['recall'])
                    precision_ma += tp_ma / k

                    tp_ma_ex = 0
                    ma_ex_results = self.get_top_k_indices(k, result['ma_ex'])
                    for ma_ex_result in ma_ex_results:
                        if int(ma_ex_result) in result['recall']:
                            tp_ma_ex += 1
                    recall_ma_ex += tp_ma_ex / len(result['recall'])
                    precision_ma_ex += tp_ma_ex / k

                    tp_mb = 0
                    mb_results = self.get_top_k_indices(k, result['mb'])
                    for mb_result in mb_results:
                        if int(mb_result) in result['recall']:
                            tp_mb += 1
                    recall_mb += tp_mb / len(result['recall'])
                    precision_mb += tp_mb / k

                    tp_mb_ex = 0
                    mb_ex_results = self.get_top_k_indices(k, result['mb_ex'])
                    for mb_ex_result in mb_ex_results:
                        if int(mb_ex_result) in result['recall']:
                            tp_mb_ex += 1
                    recall_mb_ex += tp_mb_ex / len(result['recall'])
                    precision_mb_ex += tp_mb_ex / k

                    tp_rag = 0
                    rag_results = self.get_top_k_indices(k, result['rag'])
                    for rag_result in rag_results:
                        if int(rag_result) in result['recall']:
                            tp_rag += 1
                    recall_rag += tp_rag / len(result['recall'])
                    precision_rag += tp_rag / k

                # グループ内のresultの数で平均をとる
                n = len(group)
                recall_sr /= n
                recall_ma /= n
                recall_ma_ex /= n
                recall_mb /= n
                recall_mb_ex /= n
                recall_rag /= n
                precision_sr /= n
                precision_ma /= n
                precision_ma_ex /= n
                precision_mb /= n
                precision_mb_ex /= n
                precision_rag /= n

                group_results.append({
                    'k': k,
                    'recall_SynapticRAG': recall_sr,
                    'recall_RAG': recall_rag,
                    'recall_MyAgent': recall_ma,
                    'recall_MyAgent(ext)': recall_ma_ex,
                    'recall_MemoryBank': recall_mb,
                    'recall_MemoryBank(ext)': recall_mb_ex,
                    'precision_SynapticRAG': precision_sr,
                    'precision_RAG': precision_rag,
                    'precision_MyAgent': precision_ma,
                    'precision_MyAgent(ext)': precision_ma_ex,
                    'precision_MemoryBank': precision_mb,
                    'precision_MemoryBank(ext)': precision_mb_ex
                })

                # いずれかの手法でrecallが0.99以上になった場合は、kループを終了
                if (recall_sr >= 0.99 or recall_ma >= 0.99 or recall_ma_ex >= 0.99 or 
                    recall_mb >= 0.99 or recall_mb_ex >= 0.99 or recall_rag >= 0.99):
                    break
            
            # キーを文字列にしておく（JSONのキーは文字列になるため）
            k_results_by_group[str(recall_len)] = group_results

        # グループごとの結果をk_result.jsonに保存
        with open(self.result_dir + '/k_result.json', 'w') as f:
            json.dump(k_results_by_group, f, indent=4)


    def plot(self):
        data_path = self.result_dir + '/k_result.json'
        dataset_name = os.path.basename(self.result_dir)
        with open(data_path, 'r') as f:
            data = json.load(f)

        # min_x = 1000
        min_x = 3
        max_x = 0
        for d in data:
            # min_x = min(min_x, d['k'])
            max_x = max(max_x, d['k'])
        df = pd.DataFrame(data)

        plt.rcParams['font.size'] = 24
        plt.rcParams['axes.labelsize'] = 28
        plt.rcParams['axes.titlesize'] = 28
        plt.rcParams['legend.fontsize'] = 22

        fig_recall = plt.figure(figsize=(12, 10))
        ax1 = fig_recall.add_subplot(111)
        
        for spine in ax1.spines.values():
            spine.set_linewidth(3.5)

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        markers = ['o', 's', '^', 'D', 'v', 'p']
        
        recall_columns = [col for col in df.columns if col.startswith('recall_')]
        for i, col in enumerate(recall_columns):
            ax1.plot(df['k'], df[col], marker=markers[i], label=col.replace('recall_', ''), 
                    color=colors[i], linewidth=2, markersize=8)

        scale_recall = 0.2
        ax1.set_xlabel('Top K', labelpad=10)
        ax1.set_ylabel('Average Recall @ K', labelpad=10)
        ax1.set_title(f'Recall @ K in {dataset_name}', pad=10)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(loc='lower right')
        ax1.set_xlim(min_x - 0.5, max_x + 0.5)
        ax1.set_ylim(0, 1.0)
        ax1.yaxis.set_tick_params(length=0)
        ax1.xaxis.set_tick_params(length=0)
        ax1.xaxis.set_tick_params(pad=10)
        ax1.yaxis.set_tick_params(pad=10) 
        ax1.set_xticks(range(min_x, max_x + 1))
        ax1.set_yticks(numpy.arange(0, 1.0 + scale_recall, scale_recall))

        plt.tight_layout()
        recall_save_path = data_path.replace('_result.json', '_recall.png')
        fig_recall.savefig(recall_save_path, bbox_inches='tight', dpi=300)
        plt.close(fig_recall)

        fig_precision = plt.figure(figsize=(12, 10))
        ax2 = fig_precision.add_subplot(111)
        
        for spine in ax2.spines.values():
            spine.set_linewidth(3.5) 

        max_y = 0
        precision_columns = [col for col in df.columns if col.startswith('precision_')]
        for i, col in enumerate(precision_columns):
            ax2.plot(df['k'], df[col], marker=markers[i], label=col.replace('precision_', ''), 
                    color=colors[i], linewidth=2, markersize=8)
            max_y = max(max_y, max(df[col]))

        scale_precision = 0.05
        ax2.set_xlabel('Top K', labelpad=10)
        ax2.set_ylabel('Average Precision @ K', labelpad=10)
        ax2.set_title(f'Precision @ K in {dataset_name}', pad=10)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.yaxis.set_tick_params(length=0)
        ax2.xaxis.set_tick_params(length=0)
        ax2.xaxis.set_tick_params(pad=10)
        ax2.yaxis.set_tick_params(pad=10)         
        ax2.legend(loc='upper right')
        ax2.set_xlim(min_x - 0.5, max_x + 0.5)
        ax2.set_ylim(0, max_y + scale_precision)
        ax2.set_xticks(range(min_x, max_x + 1))
        ax2.set_yticks(numpy.arange(0, max_y + scale_precision, scale_precision))

        plt.tight_layout()
        precision_save_path = data_path.replace('_result.json', '_precision.png')
        fig_precision.savefig(precision_save_path, bbox_inches='tight', dpi=300)
        plt.close(fig_precision)

    def get_top_k_indices(self, k, results):
        sorted_items = sorted(results.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_items[:k])
    
    def prepare_dialogue(self, dialogue, faiss, sr_sql, ma_sql, ma_ex_sql, mb_sql, mb_ex_sql, sr_params, ma_params, ma_ex_params, mb_ex_params, measure_time=None):
        cos_th = sr_params['cos_th']
        v_th = sr_params['v_th']
        stimulus_th = sr_params['stimulus_th']
        tau_init = sr_params['tau_init']
        tau_scale = sr_params['tau_scale']
        time_scale = sr_params['time_scale']
        bond_scale = sr_params['bond_scale']
        v_rest = sr_params['v_rest']
        i_rest = sr_params['i_rest']

        ma_cos_th = ma_params['cos_th']

        ma_ex_cos_th = ma_ex_params['cos_th']
        r_scale = ma_ex_params['r_scale']
        t_scale = ma_ex_params['t_scale']
        g_scale = ma_ex_params['g_scale']

        top_k = mb_ex_params['top_k']
        forget_th = mb_ex_params['forget_th']
        mb_t_scale = mb_ex_params['t_scale']
        s_scale = mb_ex_params['s_scale']
        s_init = mb_ex_params['s_init']

        sr_data = self.prepare_data(faiss, sr_sql, dialogue['index'], dialogue['vector'])
        ma_data = self.prepare_data(None, ma_sql, dialogue['index'], dialogue['vector'])
        ma_ex_data = self.prepare_data(None, ma_ex_sql, dialogue['index'], dialogue['vector'])
        mb_data = self.prepare_data(None, mb_sql, dialogue['index'], dialogue['vector'])
        mb_ex_data = self.prepare_data(None, mb_ex_sql, dialogue['index'], dialogue['vector'])
    
        sr_results = self.propagate_stimuli(faiss, sr_sql, sr_data, cos_th, v_th, stimulus_th, tau_init, tau_scale, time_scale, bond_scale, v_rest, i_rest)
        ma_results = self.my_agent(faiss, ma_sql, ma_data, ma_cos_th)
        ma_ex_results = self.my_agent_ex(faiss, ma_ex_sql, ma_ex_data, ma_ex_cos_th, r_scale, t_scale, g_scale)
        mb_results = self.memory_bank(faiss, mb_sql, mb_data)
        mb_ex_results = self.memory_bank_ex(faiss, mb_ex_sql, mb_ex_data, top_k, forget_th, mb_t_scale, s_scale, s_init)
        return sr_results, ma_results, ma_ex_results, mb_results, mb_ex_results

    def prepare_data(self, faiss, sql, index: int, vector: list) -> dict:
        data =\
        {
            'id': index,
            'vector': vector,
            'message': '',
            'fire': -1,
            'v': 0,
            'i': 0,
            'tau': 1,
            'spike': [[1], [index]]
        }
        if faiss is not None:
            faiss.add_embeddings(embedding=data['vector'], id=index)
        sql.add_data_to_database(data)
        return data
    
    @average_timer
    def propagate_stimuli(self, faiss, sql, initial_data: dict, cos_th, v_th, stimulus_th, tau_init, tau_scale, time_scale, bond_scale, v_rest, i_rest) -> dict:
        parent_queue = [(initial_data, 1)]
        queried_indices = []
        memories = {}
        generation = 0
        while parent_queue:
            generation += 1
            self.log.info(words.get('processing_generation', generation=generation))
            child_queue = []
            children_data = {}
            for parent_data, _ in parent_queue:
                queried_indices.append(parent_data['id'])
            for parent_data, parent_stimulus in parent_queue:
                parent_vector = numpy.array(parent_data['vector'])
                results = faiss.search_embeddings(query_vector=parent_vector, threshold=cos_th, exclude_ids=queried_indices)
                self.log.debug(words.get('search_results', results=results))
                for index, distance in zip(*results):
                    child_data = sql.get_data_by_index(index)
                    child_stimulus = self.memory.stimulate(distance=distance, parent_data=parent_data, child_data=child_data, tau_init=tau_init, tau_scale=tau_scale, bond_scale=bond_scale)
                    stimulus = parent_stimulus * child_stimulus
                    if stimulus < stimulus_th:
                        continue
                    if child_data['id'] in children_data and children_data[child_data['id']][1] > stimulus:
                        continue
                    copied_data = child_data.copy()
                    copied_data['spike'][0].append(stimulus)
                    copied_data['spike'][1].append(initial_data['id'])
                    lif_data = self.memory.lif(data=copied_data, count=initial_data['id'], v_th=v_th, tau_init=tau_init, tau_scale=tau_scale, time_scale=time_scale, v_rest=v_rest, i_rest=i_rest)
                    for key in lif_data:
                        copied_data[key] = lif_data[key]
                    children_data[child_data['id']] = (copied_data, stimulus)
            for _, (data, stim) in children_data.items():
                sql.update_data_in_database(data)
                if data['fire'] != -1:
                    memories[data['id']] = data['fire'] + 1000
                else:
                    memories[data['id']] = data['v'] + 100
                child_queue.append((data, stim))
            parent_queue = child_queue
            child_queue = []
        return memories

    # @average_timer
    # def propagate_stimuli(self, faiss, sql, initial_data: dict, cos_th, v_th, stimulus_th, tau_init, tau_scale, time_scale, bond_scale, v_rest, i_rest) -> dict:
    #     parent_queue = [(initial_data, 1)]
    #     queried_indices = []
    #     memories = {}
    #     generation = 0
    #     while parent_queue:
    #         generation += 1
    #         self.log.info(words.get('processing_generation', generation=generation))
    #         child_queue = []
    #         children_data = {}
    #         for parent_data, _ in parent_queue:
    #             queried_indices.append(parent_data['id'])

    #         center_parent_vector = numpy.zeros(len(parent_queue[0][0]['vector']))
    #         for parent_data, parent_stimulus in parent_queue:
    #             parent_vector = numpy.array(parent_data['vector'])
    #             # center_parent_vector = numpy.add(center_parent_vector, parent_vector)
    #             center_parent_vector += parent_vector
    #             # print(parent_vector)
    #         center_parent_vector /= len(parent_queue)
    #         # max_cos_similarity = 0
    #         # for parent_data, parent_stimulus in parent_queue:
    #         #     max_cos_similarity = max(max_cos_similarity, numpy.dot(center_parent_vector, parent_data['vector']))


    #         results = faiss.search_embeddings(query_vector=center_parent_vector, threshold=cos_th, exclude_ids=queried_indices, top_k=10)
    #         for parent_data, parent_stimulus in parent_queue:
    #             self.log.debug(words.get('search_results', results=results))
    #             for index, distance in zip(*results):
    #                 child_data = sql.get_data_by_index(index)
    #                 child_stimulus = self.memory.stimulate(distance=distance, parent_data=parent_data, child_data=child_data, tau_init=tau_init, tau_scale=tau_scale, bond_scale=bond_scale)
    #                 stimulus = parent_stimulus * child_stimulus
    #                 if stimulus < stimulus_th:
    #                     continue
    #                 if child_data['id'] in children_data and children_data[child_data['id']][1] > stimulus:
    #                     continue
    #                 copied_data = child_data.copy()
    #                 copied_data['spike'][0].append(stimulus)
    #                 copied_data['spike'][1].append(initial_data['id'])
    #                 lif_data = self.memory.lif(data=copied_data, count=initial_data['id'], v_th=v_th, tau_init=tau_init, tau_scale=tau_scale, time_scale=time_scale, v_rest=v_rest, i_rest=i_rest)
    #                 for key in lif_data:
    #                     copied_data[key] = lif_data[key]
    #                 children_data[child_data['id']] = (copied_data, stimulus)
    #         for _, (data, stim) in children_data.items():
    #             sql.update_data_in_database(data)
    #             if data['fire'] != -1:
    #                 memories[data['id']] = data['fire'] + 1000
    #             else:
    #                 memories[data['id']] = data['v'] + 100
    #             child_queue.append((data, stim))
    #         parent_queue = child_queue
    #         child_queue = []
    #     return memories
    
    @average_timer
    def rag(self, faiss, query_vector, exclude_ids):
        return faiss.search_embeddings(query_vector=query_vector, exclude_ids=exclude_ids)

    @average_timer
    def my_agent(self, faiss, sql, data, cos_th):
        def calc_p(input_data: dict, memory_data: dict, distance: float) -> float:
            r = distance
            t = input_data['spike'][1][-1] - memory_data['spike'][1][-1]
            g = memory_data['tau']
            if g == 0:
                return 0
            p = (1 - numpy.exp((-r) * numpy.exp(-t / g))) / (1 - numpy.exp(-1))
            return p

        def calc_g(data: dict, count) -> float:
            g = data['tau']
            t = count - data['spike'][1][-1]
            g = g + (1 - numpy.exp(-t)) / (1 + numpy.exp(-t))
            return g
        
        results = faiss.search_embeddings(query_vector=data['vector'], exclude_ids=[data['id']])
        max_p = 0
        fired_memory = None
        memory_p = {}
        for index, distance in zip(*results):
            memory = sql.get_data_by_index(index)
            g = calc_g(data=memory, count=data['id'])
            memory['tau'] = g
            p = calc_p(input_data=data, memory_data=memory, distance=distance)
            if distance >= cos_th:
                memory_p[index] = p + 1
            else:
                memory_p[index] = p
            if distance >= cos_th and p > max_p:
                max_p = p
                fired_memory = memory
        if fired_memory:
            fired_memory['spike'][1][-1] = data['id']
            sql.update_data_in_database(fired_memory)
        return memory_p
    
    @average_timer
    def my_agent_ex(self, faiss, sql, data, cos_th, r_scale, t_scale, g_scale):
        def calc_p(input_data: dict, memory_data: dict, distance: float) -> float:
            r = distance
            r *= r_scale
            t = input_data['spike'][1][-1] - memory_data['spike'][1][-1]
            t *= t_scale
            g = memory_data['tau']
            g *= g_scale
            if g == 0:
                return 0
            p = (1 - numpy.exp((-r) * numpy.exp(-t / g))) / (1 - numpy.exp(-1))
            return p

        def calc_g(data: dict, count) -> float:
            g = data['tau']
            t = count - data['spike'][1][-1]
            t *= t_scale
            g = g + (1 - numpy.exp(-t)) / (1 + numpy.exp(-t))
            return g
        
        results = faiss.search_embeddings(query_vector=data['vector'], exclude_ids=[data['id']])
        max_p = 0
        fired_memory = None
        memory_p = {}
        for index, distance in zip(*results):
            memory = sql.get_data_by_index(index)
            g = calc_g(data=memory, count=data['id'])
            memory['tau'] = g
            p = calc_p(input_data=data, memory_data=memory, distance=distance)
            if distance >= cos_th:
                memory_p[index] = p + 1
            else:
                memory_p[index] = p
            if distance > cos_th and p > max_p:
                max_p = p
                fired_memory = memory
        if fired_memory:
            fired_memory['spike'][1][-1] = data['id']
            sql.update_data_in_database(fired_memory) 
        return memory_p
    
    @average_timer
    def memory_bank(self, faiss, sql, data):
        results = faiss.search_embeddings(query_vector=data['vector'], exclude_ids=[data['id']])
        recall_memories = [[], []]
        memories = {}
        for index, distance in zip(*results):
            memory = sql.get_data_by_index(index)
            s = memory['tau']
            t = data['id'] - memory['spike'][1][-1]
            score = numpy.exp(-t / 5*s)
            memories[memory['id']] = score
            if random.random() <= score:
                recall_memories[0].append(memory)
                recall_memories[1].append(score)
            if len(recall_memories[0]) > 0 and len(recall_memories[1]) > 0:
                recall_memories = self.get_top_n_pairs(recall_memories[0], recall_memories[1], 6)
            else:
                recall_memories = [[], []]
            for memory, score in zip(*recall_memories):
                memory['spike'][1][-1] = data['id']
                memory['tau'] = memory['tau'] + 1
                sql.update_data_in_database(memory)
        return memories
    
    @average_timer
    def memory_bank_ex(self, faiss, sql, data, top_k, forget_th, t_scale, s_scale, s_init):
        results = faiss.search_embeddings(query_vector=data['vector'], exclude_ids=[data['id']])
        recall_memories = [[], []]
        memories = {}
        for index, distance in zip(*results):
            memory = sql.get_data_by_index(index)
            s = memory['tau'] + s_init
            s *= s_scale
            if s_scale == 0:
                return memories
            t = data['id'] - memory['spike'][1][-1]
            t *= t_scale
            score = numpy.exp(-t / s)
            memories[memory['id']] = score
            if forget_th <= score:
                recall_memories[0].append(memory)
                recall_memories[1].append(score)
        if len(recall_memories[0]) > 0 and len(recall_memories[1]) > 0:
            recall_memories = self.get_top_n_pairs(recall_memories[0], recall_memories[1], top_k)
        else:
            recall_memories = [[], []]
        for memory, score in zip(*recall_memories):
            memory['spike'][1][-1] = data['id']
            memory['tau'] = memory['tau'] + 1
            sql.update_data_in_database(memory)
        return memories

    def read_json_data(self, path):
        with open(path, 'r', encoding="utf-8") as f:
            content = f.read()
            dataset = json.loads(content)
        return dataset

    def clear_directory(self):
        if os.path.exists(self.conv_dir):
            for filename in os.listdir(self.conv_dir):
                file_path = os.path.join(self.conv_dir, filename)
                try:
                    if filename == '.gitkeep':
                        continue
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    self.log.error(f'Failed to delete {file_path}. Reason: {e}')
        else:
            self.log.error(f"Characters directory {self.conv_dir} does not exist.")
    
    def get_top_n_pairs(self, indices, distances, n):
        sorted_pairs = sorted(zip(indices, distances), key=lambda x: x[1], reverse=True)
        top_n_pairs = sorted_pairs[:n]
        top_n_indices = [pair[0] for pair in top_n_pairs]
        top_n_distances = [pair[1] for pair in top_n_pairs]
        return [top_n_indices, top_n_distances]