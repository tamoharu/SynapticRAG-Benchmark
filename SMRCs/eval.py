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
        self.conv_log_path_1 = os.path.abspath(os.path.join(os.path.dirname(__file__), './log/conv_log_1.txt'))
        self.conv_log_path_2 = os.path.abspath(os.path.join(os.path.dirname(__file__), './log/conv_log_2.txt'))
        self.conv_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../process_files'))
        self.count = 0
        if lang == 'en':
            self.dialogue_path = os.path.abspath(os.path.join(os.path.dirname(__file__), './dataset/SMRCs/dataset_EN.json'))
            self.db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), './dataset/prepared/vectors_en.db'))
        elif lang == 'ja':
            self.dialogue_path = os.path.abspath(os.path.join(os.path.dirname(__file__), './dataset/SMRCs/dataset_JA.json'))
            self.db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), './dataset/prepared/vectors_ja.db'))
        else:
            raise ValueError(f"Unsupported language: {lang}")

    def eval_run(self, sr_params, ma_params, ma_ex_params, mb_ex_params):
        self.clear_directory()
        sr = 0
        ma_1 = 0
        ma_2 = 0
        ma_ex_1 = 0
        ma_ex_2 = 0
        mb_1 = 0
        mb_2 = 0
        mb_ex_1 = 0
        mb_ex_2 = 0
        rag_1 = 0
        rag_2 = 0
        tasks = 0
        dataset = self.read_json_data(self.dialogue_path)
        section_count = len(dataset)

        def process_section(section):
            nonlocal sr, ma_1, ma_2, ma_ex_1, ma_ex_2, mb_1, mb_2, mb_ex_1, mb_ex_2, rag_1, rag_2, tasks
            faiss = Faiss(dimension=3072, index_path=f'section_{section}.index')
            sr_sql = SQL(f'section_{section}.db', "SynapticRAG")
            ma_sql = SQL(f'section_{section}.db', "MyAent")
            ma_ex_sql = SQL(f'section_{section}.db', "MyAgent_ex")
            mb_sql = SQL(f'section_{section}.db', "MemoryBank")
            mb_ex_sql = SQL(f'section_{section}.db', "MemoryBank_ex")
            dialogues = self.get_dialogues(section)

            local_sr = 0
            local_ma_1 = 0
            local_ma_2 = 0
            local_ma_ex_1 = 0
            local_ma_ex_2 = 0
            local_mb_1 = 0
            local_mb_2 = 0
            local_mb_ex_1 = 0
            local_mb_ex_2 = 0
            local_rag_1 = 0
            local_rag_2 = 0
            local_tasks = 0

            for dialogue in dialogues:
                sr_results, ma_results, ma_ex_results, mb_results, mb_ex_results = self.prepare_dialogue(dialogue, faiss, sr_sql, ma_sql, ma_ex_sql, mb_sql, mb_ex_sql, sr_params, ma_params, ma_ex_params, mb_ex_params)
                recall_indices = dialogue['recall']
                if len(recall_indices) > 0:
                    local_tasks += len(recall_indices)
                    for result in sr_results:
                        if result in recall_indices:
                            local_sr += 1
                    ma_results_1 = self.get_top_n_indices(ma_results[0], ma_results[1], len(sr_results))
                    for result in ma_results_1:
                        if result in recall_indices:
                            local_ma_1 += 1

                    ma_ex_results_1 = self.get_top_n_indices(ma_ex_results[0], ma_ex_results[1], len(sr_results))
                    for result in ma_ex_results_1:
                        if result in recall_indices:
                            local_ma_ex_1 += 1

                    mb_results_1 = self.get_top_n_indices(mb_results[0], mb_results[1], len(sr_results))
                    for result in mb_results_1:
                        if result in recall_indices:
                            local_mb_1 += 1

                    mb_ex_results_1 = self.get_top_n_indices(mb_ex_results[0], mb_ex_results[1], len(sr_results))
                    for result in mb_ex_results_1:
                        if result in recall_indices:
                            local_mb_ex_1 += 1

                    rag_search = faiss.search_embeddings(query_vector=dialogue['vector'], exclude_ids=[dialogue['index']])
                    rag_results_1 = self.get_top_n_indices(rag_search[0], rag_search[1], len(sr_results))
                    for result in rag_results_1:
                        if result in recall_indices:
                            local_rag_1 += 1

                    if len(sr_results) >= len(recall_indices):
                        rag_results_2 = self.get_top_n_indices(rag_search[0], rag_search[1], len(sr_results))
                        ma_results_2 = self.get_top_n_indices(ma_results[0], ma_results[1], len(sr_results))
                        ma_ex_results_2 = self.get_top_n_indices(ma_ex_results[0], ma_ex_results[1], len(sr_results))
                        mb_results_2 = self.get_top_n_indices(mb_results[0], mb_results[1], len(sr_results))
                        mb_ex_results_2 = self.get_top_n_indices(mb_ex_results[0], mb_ex_results[1], len(sr_results))
                    else:
                        rag_results_2 = self.get_top_n_indices(rag_search[0], rag_search[1], len(recall_indices))
                        ma_results_2 = self.get_top_n_indices(ma_results[0], ma_results[1], len(recall_indices))
                        ma_ex_results_2 = self.get_top_n_indices(ma_ex_results[0], ma_ex_results[1], len(recall_indices))
                        mb_results_2 = self.get_top_n_indices(mb_results[0], mb_results[1], len(recall_indices))
                        mb_ex_results_2 = self.get_top_n_indices(mb_ex_results[0], mb_ex_results[1], len(recall_indices))

                    for result in ma_results_2:
                        if result in recall_indices:
                            local_ma_2 += 1

                    for result in ma_ex_results_2:
                        if result in recall_indices:
                            local_ma_ex_2 += 1

                    for result in mb_results_2:
                        if result in recall_indices:
                            local_mb_2 += 1

                    for result in mb_ex_results_2:
                        if result in recall_indices:
                            local_mb_ex_2 += 1

                    for result in rag_results_2:
                        if result in recall_indices:
                            local_rag_2 += 1

                    self.result_log(self.conv_log_path_1, section, dialogue['index'], recall_indices, sr_results, rag_results_1, ma_results_1, ma_ex_results_1, mb_results_1, mb_ex_results_1)
                    self.result_log(self.conv_log_path_2, section, dialogue['index'], recall_indices, sr_results, rag_results_2, ma_results_2, ma_ex_results_2, mb_results_2, mb_ex_results_2)

            return local_sr, local_ma_1, local_ma_2, local_ma_ex_1, local_ma_ex_2, local_mb_1, local_mb_2, local_mb_ex_1, local_mb_ex_2, local_rag_1, local_rag_2, local_tasks

        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(process_section, section): section for section in range(section_count)}
            with tqdm(total=section_count) as pbar:
                for future in as_completed(futures):
                    pbar.update(1)
                    result = future.result()
                    sr += result[0]
                    ma_1 += result[1]
                    ma_2 += result[2]
                    ma_ex_1 += result[3]
                    ma_ex_2 += result[4]
                    mb_1 += result[5]
                    mb_2 += result[6]
                    mb_ex_1 += result[7]
                    mb_ex_2 += result[8]
                    rag_1 += result[9]
                    rag_2 += result[10]
                    tasks += result[11]

        model_ratio = sr * 100 / tasks
        rag_ratio_1 = rag_1 * 100 / tasks
        rag_ratio_2 = rag_2 * 100 / tasks
        ma_ratio_1 = ma_1 * 100 / tasks
        ma_ratio_2 = ma_2 * 100 / tasks
        ma_ex_ratio_1 = ma_ex_1 * 100 / tasks
        ma_ex_ratio_2 = ma_ex_2 * 100 / tasks
        mb_ratio_1 = mb_1 * 100 / tasks
        mb_ratio_2 = mb_2 * 100 / tasks
        mb_ex_ratio_1 = mb_ex_1 * 100 / tasks
        mb_ex_ratio_2 = mb_ex_2 * 100 / tasks
        print(f'\nEvaluation 1')
        print(f"Model_ratio: {model_ratio}%, RAG_ratio: {rag_ratio_1}%, MyAgent_ratio: {ma_ratio_1}%, MyAgent_ex_ratio: {ma_ex_ratio_1}%, MemoryBank_ratio: {mb_ratio_1}%, MemoryBank_ex_ratio: {mb_ex_ratio_1}%")
        print(f"Model_count: {sr}, RAG_count: {rag_1}, MyAgent_count: {ma_1}, MyAgent_ex_count: {ma_ex_1}, MemoryBank_count: {mb_1}, MemoryBank_ex_count: {mb_ex_1}")
        print(f'\nEvaluation 2')
        print(f"Model_ratio: {model_ratio}%, RAG_ratio: {rag_ratio_2}%, MyAgent_ratio: {ma_ratio_2}%, MyAgent_ex_ratio: {ma_ex_ratio_2}%, MemoryBank_ratio: {mb_ratio_2}%, MemoryBank_ex_ratio: {mb_ex_ratio_2}%")
        print(f"Model_count: {sr}, RAG_count: {rag_2}, MyAgent_count: {ma_2}, MyAgent_ex_count: {ma_ex_2}, MemoryBank_count: {mb_2}, MemoryBank_ex_count: {mb_ex_2}")
        self.clear_directory()
        return {'sr': sr, 'rag': rag_1, 'ma': ma_1, 'ma_ex': ma_ex_1, 'mb': mb_1, 'mb_ex': mb_ex_1}


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
                    sum = numpy.sum(ma_results[1])
                    if sum != 0:
                        for index, result in enumerate(ma_results[0]):
                            if result in recall_indices:
                                ma += ma_results[1][index] / sum
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
                        for index, result in enumerate(ma_ex_results[0]):
                            if result in recall_indices:
                                ma_ex += ma_ex_results[1][index] / sum
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
                    sum = numpy.sum(mb_ex_results[1])
                    if sum != 0:
                        for index, result in enumerate(mb_ex_results[0]):
                            if result in recall_indices:
                                mb_ex += mb_ex_results[1][index] / sum
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