import sys
sys.path.append('..')
import os
import json
import numpy
import sqlite3
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.sql import SQL
from utils.faiss import Faiss
from utils.eval import Eval


class EvalProcess(Eval):
    def __init__(self, lang):
        super().__init__()
        self.conv_log_path_1 = os.path.abspath(os.path.join(os.path.dirname(__file__), '../datasets/SMRCs/log/conv_log_1.txt'))
        self.conv_log_path_2 = os.path.abspath(os.path.join(os.path.dirname(__file__), '../datasets/SMRCs/log/conv_log_2.txt'))
        self.conv_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../process_files'))
        self.count = 0
        if lang == 'en':
            self.dialogue_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dataset/SMRCs/dataset_EN.json'))
            self.db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../datasets/SMRCs/dataset/prepared/vectors_en.db'))
            self.result_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../results/SMRCs (EN)/'))
        elif lang == 'ja':
            self.dialogue_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dataset/SMRCs/dataset_JA.json'))
            self.db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../datasets/SMRCs/dataset/prepared/vectors_ja.db'))
            self.result_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../results/SMRCs (JA)/'))
        else:
            raise ValueError(f"Unsupported language: {lang}")

    
    def eval_run(self, sr_params, ma_params, ma_ex_params, mb_ex_params):
        self.clear_directory()
        dataset = self.read_json_data(self.dialogue_path)
        section_count = len(dataset)

        def process_section(section):
            faiss = Faiss(dimension=3072, index_path=f'section_{section}.index')
            sr_sql = SQL(f'section_{section}.db', "SynapticRAG")
            ma_sql = SQL(f'section_{section}.db', "MyAent")
            ma_ex_sql = SQL(f'section_{section}.db', "MyAgent_ex")
            mb_sql = SQL(f'section_{section}.db', "MemoryBank")
            mb_ex_sql = SQL(f'section_{section}.db', "MemoryBank_ex")
            dialogues = self.get_dialogues(section)
            results = []
            for dialogue in dialogues:
                sr_results, ma_results, ma_ex_results, mb_results, mb_ex_results = self.prepare_dialogue(dialogue, faiss, sr_sql, ma_sql, ma_ex_sql, mb_sql, mb_ex_sql, sr_params, ma_params, ma_ex_params, mb_ex_params)
                if len(dialogue['recall']) > 0:
                    rag_search = faiss.search_embeddings(query_vector=dialogue['vector'], exclude_ids=[dialogue['index']])
                    rag_results = {}
                    for id, distance in zip(rag_search[0], rag_search[1]):
                        rag_results[id] = distance
                    sr_memories = sr_sql.get_all_data()
                    for memory in sr_memories:
                        if memory['id'] == dialogue['index']:
                            continue
                        if memory['id'] in sr_results:
                            continue
                        sr_results[memory['id']] = memory['v']
                    result = {}
                    result['index'] = str(section) + '_' + str(dialogue['index'])
                    result['recall'] = dialogue['recall']
                    result['sr'] = sr_results
                    result['ma'] = ma_results
                    result['ma_ex'] = ma_ex_results
                    result['mb'] = mb_results
                    result['mb_ex'] = mb_ex_results   
                    result['rag'] = rag_results
                    results.append(result)
            return results
        all_results = []
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(process_section, section): section for section in range(section_count)}
            with tqdm(total=section_count) as pbar:
                for future in as_completed(futures):
                    pbar.update(1)
                    result = future.result()
                    all_results.extend(result)
        with open(self.result_dir + '/memories.json', 'w') as f:
            json.dump(all_results, f, indent=4)
        self.clear_directory()

    def sr_opt_run(self, trial, count_th, sr_params):
        cos_th = sr_params['cos_th']
        v_th = sr_params['v_th']
        stimulus_th = sr_params['stimulus_th']
        tau_init = sr_params['tau_init']
        tau_scale = sr_params['tau_scale']
        time_scale = sr_params['time_scale']
        bond_scale = sr_params['bond_scale']
        v_rest = sr_params['v_rest']
        i_rest = sr_params['i_rest']

        sample_count = 0
        sr = 0
        dataset = self.read_json_data(self.dialogue_path)
        section_count = len(dataset)
        for section in range(section_count):
            if sample_count == count_th:
                break
            sample_count += 1
            sr_sql = SQL(f'section_{section}_{trial}.db', "SynapticRAG_opt")
            faiss = Faiss(dimension=3072, index_path=f'section_{section}_{trial}.index')
            dialogues = self.get_dialogues(section)
            for dialogue in dialogues:
                recall_indices = dialogue['recall']
                data = self.prepare_data(faiss, sr_sql, dialogue['index'], dialogue['vector'])
                sr_results = self.propagate_stimuli(faiss, sr_sql, data, cos_th, v_th, stimulus_th, tau_init, tau_scale, time_scale, bond_scale, v_rest, i_rest)
                if len(recall_indices) > 0:
                    set_1 = set(sr_results.keys())
                    set_2 = set(recall_indices)
                    intersection = set_1.intersection(set_2)
                    if len(intersection) < len(recall_indices):
                        sr -= (len(recall_indices) - len(intersection)) / len(recall_indices)
                    penalty = len(sr_results) / len(recall_indices)
                    sr -= penalty * 0.15
        return sr
    
    def ma_opt_run(self, trial, count_th, ma_params):
        ma_cos_th = ma_params['cos_th']

        sample_count = 0
        ma = 0
        dataset = self.read_json_data(self.dialogue_path)
        section_count = len(dataset)
        for section in range(section_count):
            if sample_count == count_th:
                break
            sample_count += 1
            ma_sql = SQL(f'section_{section}_{trial}.db', "MyAgent_opt")
            faiss = Faiss(dimension=3072, index_path=f'section_{section}_{trial}.index')
            dialogues = self.get_dialogues(section)
            for dialogue in dialogues:
                recall_indices = dialogue['recall']
                ma_data = self.prepare_data(faiss, ma_sql, dialogue['index'], dialogue['vector'])
                ma_results = self.my_agent(faiss, ma_sql, ma_data, ma_cos_th)
                if len(recall_indices) > 0:
                    sum = numpy.sum(list(ma_results.values()))
                    if sum != 0:
                        for result in ma_results.items():
                            if int(result[0]) in recall_indices:
                                ma += result[1] / sum
        return ma
    
    def ma_ex_opt_run(self, trial, count_th, ma_ex_params):
        ma_ex_cos_th = ma_ex_params['cos_th']
        r_scale = ma_ex_params['r_scale']
        t_scale = ma_ex_params['t_scale']
        g_scale = ma_ex_params['g_scale']

        sample_count = 0
        ma_ex = 0
        dataset = self.read_json_data(self.dialogue_path)
        section_count = len(dataset)
        for section in range(section_count):
            if sample_count == count_th:
                break
            sample_count += 1
            ma_ex_sql = SQL(f'section_{section}_{trial}.db', "MyAgent_ex_opt")
            faiss = Faiss(dimension=3072, index_path=f'section_{section}_{trial}.index')
            dialogues = self.get_dialogues(section)
            for dialogue in dialogues:
                recall_indices = dialogue['recall']
                ma_ex_data = self.prepare_data(faiss, ma_ex_sql, dialogue['index'], dialogue['vector'])
                ma_ex_results = self.my_agent_ex(faiss, ma_ex_sql, ma_ex_data, ma_ex_cos_th, r_scale, t_scale, g_scale)
                if len(recall_indices) > 0:
                    sum = numpy.sum(ma_ex_results[1])
                    if sum != 0:
                        for result in ma_ex_results.items():
                            if int(result[0]) in recall_indices:
                                ma_ex += result[1] / sum
        return ma_ex
    
    def mb_ex_opt_run(self, trial, count_th, mb_ex_params):
        top_k = mb_ex_params['top_k']
        forget_th = mb_ex_params['forget_th']
        t_scale = mb_ex_params['t_scale']
        s_scale = mb_ex_params['s_scale']
        s_init = mb_ex_params['s_init']

        sample_count = 0
        mb_ex = 0
        dataset = self.read_json_data(self.dialogue_path)
        section_count = len(dataset)
        for section in range(section_count):
            if sample_count == count_th:
                break
            sample_count += 1
            mb_ex_sql = SQL(f'section_{section}_{trial}.db', "MemoryBank_ex_opt")
            faiss = Faiss(dimension=3072, index_path=f'section_{section}_{trial}.index')
            dialogues = self.get_dialogues(section)
            for dialogue in dialogues:
                recall_indices = dialogue['recall']
                mb_ex_data = self.prepare_data(faiss, mb_ex_sql, dialogue['index'], dialogue['vector'])
                mb_ex_results = self.memory_bank_ex(faiss, mb_ex_sql, mb_ex_data, top_k, forget_th, t_scale, s_scale, s_init)
                if len(recall_indices) > 0:
                    sum = numpy.sum(list(mb_ex_results.values()))
                    if sum != 0:
                        for result in mb_ex_results.items():
                            if int(result[0]) in recall_indices:
                                mb_ex += result[1] / sum
        return mb_ex
    
    def get_dialogues(self, section):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        query = """
        SELECT text_index, text, recall, vector
        FROM vectors
        WHERE section = ?
        ORDER BY text_index
        """
        cursor.execute(query, (section,))
        results = cursor.fetchall()
        dialogues = []
        for index, text, recall, vector in results:
            dialogues.append({
                "index": index,
                "text": text,
                "recall": json.loads(recall),
                "vector": json.loads(vector)
            })
        dialogues = sorted(dialogues, key=lambda x: x['index'])
        conn.close()
        return dialogues