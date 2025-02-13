import os
import json
import numpy

from utils.eval import Eval


class Ablation(Eval):
    def __init__(self):
        super().__init__()
        self.memory = MemoryModel()
        self.ablation_dir = ''
        self.result_dir = ''
    
    def propagate_stimuli_ab(self, faiss, sql, initial_data: dict, cos_th, v_th, stimulus_th, tau_init, tau_scale, time_scale, bond_scale, v_rest, i_rest, ab_stim=False, ab_lif=False, ab_prop=False) -> dict:
        if ab_prop:
            stimulus_th = 0
        parent_queue = [(initial_data, 1)]
        queried_indices = []
        memories = {}
        generation = 0
        while parent_queue:
            generation += 1
            child_queue = []
            children_data = {}
            for parent_data, _ in parent_queue:
                queried_indices.append(parent_data['id'])
            for parent_data, parent_stimulus in parent_queue:
                parent_vector = numpy.array(parent_data['vector'])
                results = faiss.search_embeddings(query_vector=parent_vector, threshold=cos_th, exclude_ids=queried_indices)
                for index, distance in zip(*results):
                    child_data = sql.get_data_by_index(index)
                    child_stimulus = None
                    if ab_stim:
                        child_stimulus = self.memory.stimulate(distance=distance, parent_data=parent_data, child_data=child_data, tau_init=tau_init, tau_scale=tau_scale, bond_scale=bond_scale, ablation=True)
                    else:
                        child_stimulus = self.memory.stimulate(distance=distance, parent_data=parent_data, child_data=child_data, tau_init=tau_init, tau_scale=tau_scale, bond_scale=bond_scale)
                    stimulus = parent_stimulus * child_stimulus
                    if stimulus < stimulus_th:
                        continue
                    if child_data['id'] in children_data and children_data[child_data['id']][1] > stimulus:
                        continue
                    copied_data = child_data.copy()
                    copied_data['spike'][0].append(stimulus)
                    copied_data['spike'][1].append(initial_data['id'])
                    lif_data = None
                    if ab_lif:
                        lif_data = self.memory.lif(data=copied_data, count=initial_data['id'], v_th=v_th, tau_init=tau_init, tau_scale=tau_scale, time_scale=time_scale, v_rest=v_rest, i_rest=i_rest, ablation=True)
                    else:
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
    
    def post(self):
        param_files = os.listdir(self.ablation_dir)
        all_results = {}
        for param_file in param_files:
            if not param_file.endswith('.json'):
                continue
            param_name = param_file.split('.')[0]
            path = self.ablation_dir + '/' + param_file
            min_diff, sr_count = self.calc_score(path)
            results = {
                'min_diff': min_diff,
                'sr_count': sr_count
            }
            all_results[param_name] = results
        with open(self.result_dir + '/ablation.json', 'w') as f:
            json.dump(all_results, f, indent=4)

    def calc_score(self, path):
        task_count = 0
        sr_count = 0
        rag_count = 0
        ma_count = 0
        ma_ex_count = 0
        mb_count = 0
        mb_ex_count = 0
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

            ma_results = self.get_top_k_indices(len(sr_results), result['ma'])
            for ma_result in ma_results:
                if int(ma_result) in result['recall']:
                    ma_count += 1

            mb_results = self.get_top_k_indices(len(sr_results), result['mb'])
            for mb_result in mb_results:
                if int(mb_result) in result['recall']:
                    mb_count += 1

            ma_ex_results = self.get_top_k_indices(len(sr_results), result['ma_ex'])
            for ma_ex_result in ma_ex_results:
                if int(ma_ex_result) in result['recall']:
                    ma_ex_count += 1

            mb_ex_results = self.get_top_k_indices(len(sr_results), result['mb_ex'])
            for mb_ex_result in mb_ex_results:
                if int(mb_ex_result) in result['recall']:
                    mb_ex_count += 1
            
            min_diff = 1000
            for result in [rag_count, ma_count, mb_count, ma_ex_count, mb_ex_count]:
                min_diff = min(min_diff, sr_count - result)
            
            min_diff /= task_count
        return min_diff, sr_count


class MemoryModel:
    def __init__(self):
        self.dt = 1

    def lif(self, data, count, v_th, tau_init, time_scale, tau_scale, v_rest, i_rest, ablation=False):
        v_init: float = data['v']
        i_init: float = data['i']
        tau: float = data['tau']
        spike: list[list[float]] = data['spike']
        weights, times = spike
        elapsed_time = times[-1] - times[-2]
        t = numpy.arange(0, elapsed_time + 1, self.dt)
        v = numpy.zeros_like(t, dtype=float)
        i = numpy.zeros_like(t, dtype=float)
        s = numpy.zeros_like(t, dtype=float)
        v[0] = v_init
        i[0] = i_init
        s[0] = weights[-1]
        elapsed_time = elapsed_time * time_scale
        if ablation:
            tau = tau_init * tau_scale
            mem_tau = tau
        else:
            mem_tau = tau + (1 - numpy.exp(-elapsed_time)) / (1 + numpy.exp(-elapsed_time))
            tau = (mem_tau + tau_init) * tau_scale
        for j in range(1, len(t)):
            di = (-i[j-1] + s[j-1] - i_rest) * self.dt / tau
            i[j] = i[j-1] + di
            dv = (-(v[j-1] - v_rest) + i[j]) * self.dt / tau
            v[j] = v[j-1] + dv
            if v[j] >= v_th:
                return {'fire': v[j], 'v': v_rest, 'i': i_rest, 'tau': mem_tau, 'spike': [[1], [count]]}
        return {'fire': -1, 'v': v[-1], 'i': i[-1], 'tau': mem_tau, 'spike': spike}

    def stimulate(self, distance, parent_data, child_data, tau_init, tau_scale, bond_scale, ablation=False):
        tau = (0.5 * (parent_data['tau'] + child_data['tau']) + tau_init) * tau_scale
        _, parent_times = parent_data['spike']
        _, child_times = child_data['spike']
        stimulus = None
        if ablation:
            stimulus = distance
        else:
            bond_score = self._calculate_bond_score(parent_times, child_times, tau) * bond_scale
            stimulus = distance * bond_score
        return stimulus
    
    def _calculate_bond_score(self, A, B, tau):
        A = numpy.array(A)
        B = numpy.array(B)
        D = numpy.abs(A[:, numpy.newaxis] - B[numpy.newaxis, :])
        n, m = D.shape
        C = numpy.zeros((n, m))
        C[0, 0] = D[0, 0]
        C[1:, 0] = numpy.cumsum(D[1:, 0]) + C[0, 0]
        C[0, 1:] = numpy.cumsum(D[0, 1:]) + C[0, 0]
        for i in range(1, n):
            for j in range(1, m):
                C[i, j] = D[i, j] + min(C[i-1, j], C[i, j-1], C[i-1, j-1])
        i, j = numpy.array(C.shape) - 1
        path = [(i, j)]
        while i > 0 and j > 0:
            step_costs = [C[i-1, j], C[i, j-1], C[i-1, j-1]]
            step_index = numpy.argmin(step_costs)
            if step_index == 0:
                i -= 1
            elif step_index == 1:
                j -= 1
            else:
                i -= 1
                j -= 1
            path.append((i, j))
        while i > 0:
            i -= 1
            path.append((i, j))
        while j > 0:
            j -= 1
            path.append((i, j))
        path = path[::-1]
        score = 0
        for i in path:
            distance = abs(A[i[0]] - B[i[1]])
            score += numpy.exp(-int(distance) / tau)
        score = 1 / (1 + numpy.exp(-score))
        return score