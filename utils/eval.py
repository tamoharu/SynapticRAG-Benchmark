import os
import shutil
import json
import numpy
from config import words
from utils.log import Log
from utils.memory_model import MemoryModel


log_level = 'error'


class Eval:
    def __init__(self):
        self.log = Log(log_level=log_level)
        self.memory = MemoryModel()
        self.conv_dir = ''

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
    
        import time
        start = time.time()
        sr_results = self.propagate_stimuli(faiss, sr_sql, sr_data, cos_th, v_th, stimulus_th, tau_init, tau_scale, time_scale, bond_scale, v_rest, i_rest)
        if measure_time:
            print(f"SR: {time.time() - start}")
        start = time.time()
        ma_results = self.my_agent(faiss, ma_sql, ma_data, ma_cos_th)
        if measure_time:
            print(f"MA: {time.time() - start}")
        start = time.time()
        ma_ex_results = self.my_agent_ex(faiss, ma_ex_sql, ma_ex_data, ma_ex_cos_th, r_scale, t_scale, g_scale)
        if measure_time:
            print(f"MA_EX: {time.time() - start}")
        start = time.time()
        mb_results = self.memory_bank(faiss, mb_sql, mb_data)
        if measure_time:
            print(f"MB: {time.time() - start}")
        start = time.time()
        mb_ex_results = self.memory_bank_ex(faiss, mb_ex_sql, mb_ex_data, top_k, forget_th, mb_t_scale, s_scale, s_init)
        if measure_time:
            print(f"MB_EX: {time.time() - start}")
        return sr_results, ma_results, ma_ex_results, mb_results, mb_ex_results

    def prepare_data(self, faiss, sql, index: int, vector: list) -> dict:
        data =\
        {
            'id': index,
            'vector': vector,
            'message': '',
            'fire': False,
            'v': 0,
            'i': 0,
            'tau': 1,
            'spike': [[1], [index]]
        }
        if faiss is not None:
            faiss.add_embeddings(embedding=data['vector'], id=index)
        sql.add_data_to_database(data)
        return data

    def propagate_stimuli(self, faiss, sql, initial_data: dict, cos_th, v_th, stimulus_th, tau_init, tau_scale, time_scale, bond_scale, v_rest, i_rest) -> dict:
        parent_queue = [(initial_data, 1)]
        queried_indices = []
        fired_memories = {}
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
                if data['fire']:
                    fired_memories[data['id']] = data['v']
                child_queue.append((data, stim))
            parent_queue = child_queue
            child_queue = []
        return fired_memories
    
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
        memory_p = [[], []]
        for index, distance in zip(*results):
            memory = sql.get_data_by_index(index)
            g = calc_g(data=memory, count=data['id'])
            memory['tau'] = g
            p = calc_p(input_data=data, memory_data=memory, distance=distance)
            memory_p[0].append(index)
            memory_p[1].append(p)
            if distance >= cos_th:
                memory_p[1].append(p + 1)
            else:
                memory_p[1].append(p)
            if distance >= cos_th and p > max_p:
                max_p = p
                fired_memory = memory
        if fired_memory:
            fired_memory['spike'][1][-1] = data['id']
            sql.update_data_in_database(fired_memory)
        return memory_p
    
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
        memory_p = [[], []]
        for index, distance in zip(*results):
            memory = sql.get_data_by_index(index)
            g = calc_g(data=memory, count=data['id'])
            memory['tau'] = g
            p = calc_p(input_data=data, memory_data=memory, distance=distance)
            memory_p[0].append(index)
            if distance >= cos_th:
                memory_p[1].append(p + 1)
            else:
                memory_p[1].append(p)
            if distance > cos_th and p > max_p:
                max_p = p
                fired_memory = memory
        if fired_memory:
            fired_memory['spike'][1][-1] = data['id']
            sql.update_data_in_database(fired_memory) 
        return memory_p
    
    def memory_bank(self, faiss, sql, data):
        import random
        results = faiss.search_embeddings(query_vector=data['vector'], exclude_ids=[data['id']])
        recall_memories = [[], []]
        memories = [[], []]
        for index, distance in zip(*results):
            memory = sql.get_data_by_index(index)
            s = memory['tau']
            t = data['id'] - memory['spike'][1][-1]
            score = numpy.exp(-t / 5*s)
            memories[0].append(memory['id'])
            memories[1].append(score)
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
    
    def memory_bank_ex(self, faiss, sql, data, top_k, forget_th, t_scale, s_scale, s_init):
        results = faiss.search_embeddings(query_vector=data['vector'], exclude_ids=[data['id']])
        recall_memories = [[], []]
        memories = [[], []]
        for index, distance in zip(*results):
            memory = sql.get_data_by_index(index)
            s = memory['tau'] + s_init
            s *= s_scale
            if s_scale == 0:
                return memories
            t = data['id'] - memory['spike'][1][-1]
            t *= t_scale
            score = numpy.exp(-t / s)
            memories[0].append(memory['id'])
            memories[1].append(score)
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

    def get_top_n_indices(self, indices, distances, n):
        if not indices or not distances:
            return []
        sorted_pairs = sorted(zip(indices, distances), key=lambda x: x[1], reverse=True)
        top_n_indices = [pair[0] for pair in sorted_pairs[:n]]
        return top_n_indices
    
    def get_top_n_pairs(self, indices, distances, n):
        sorted_pairs = sorted(zip(indices, distances), key=lambda x: x[1], reverse=True)
        top_n_pairs = sorted_pairs[:n]
        top_n_indices = [pair[0] for pair in top_n_pairs]
        top_n_distances = [pair[1] for pair in top_n_pairs]
        return [top_n_indices, top_n_distances]