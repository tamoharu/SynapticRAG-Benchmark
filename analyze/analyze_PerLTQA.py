import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import sqlite3

from utils.analyze import Analyze
from utils.sql import SQL
from utils.faiss import Faiss


class AnalyseProcess(Analyze):
    def __init__(self, lang):
        super().__init__()
        self.conv_log_path_1 = os.path.abspath(os.path.join(os.path.dirname(__file__), '../datasets/PerLTQA/log/conv_log_1.txt'))
        self.conv_log_path_2 = os.path.abspath(os.path.join(os.path.dirname(__file__), '../datasets/PerLTQA/log/conv_log_2.txt'))
        self.conv_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../process_files'))
        self.count = 0
        if lang == 'en':
            self.dialogue_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../datasets/PerLTQA/dataset/prepared/filtered_dialogue.json'))
            self.qa_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../datasets/PerLTQA/dataset/prepared/filtered_qa.json'))
            self.dialogue_db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../datasets/PerLTQA/dataset/prepared/dialogue.db'))
            self.qa_db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../datasets/PerLTQA/dataset/prepared/qa.db'))
            self.result_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../results/PerLTQA (EN)/'))
            self.analyze_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../analyze/PerLTQA (EN)/'))
        elif lang == 'zh':
            self.dialogue_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../datasets/PerLTQA/dataset/prepared/filtered_dialogue_zh.json'))
            self.qa_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../datasets/PerLTQA/dataset/prepared/filtered_qa_zh.json'))
            self.dialogue_db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../datasets/PerLTQA/dataset/prepared/dialogue_zh.db'))
            self.qa_db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../datasets/PerLTQA/dataset/prepared/qa_zh.db'))
            self.result_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../results/PerLTQA (CN)/'))
            self.analyze_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../analyze/PerLTQA (CN)/'))
        else:
            raise ValueError(f"Unsupported language: {lang}")
    
    def analyze(self, sr_params, param):
        self.clear_directory()
        cos_th = sr_params['cos_th']
        v_th = sr_params['v_th']
        stimulus_th = sr_params['stimulus_th']
        tau_init = sr_params['tau_init']
        tau_scale = sr_params['tau_scale']
        time_scale = sr_params['time_scale']
        bond_scale = sr_params['bond_scale']
        v_rest = sr_params['v_rest']
        i_rest = sr_params['i_rest']
        dataset = self.read_json_data(self.dialogue_path)
        section_count = len(dataset)

        def process_section(section):
            sr_sql = SQL(f'section_{section}.db', "SynapticRAG_opt")
            faiss = Faiss(dimension=3072, index_path=f'section_{section}.index')
            dialogues = self.get_dialogues(section)
            for dialogue in dialogues:
                sr_data = self.prepare_data(faiss, sr_sql, dialogue['index'], dialogue['vector'])
                self.propagate_stimuli(faiss, sr_sql, sr_data, cos_th, v_th, stimulus_th, tau_init, tau_scale, time_scale, bond_scale, v_rest, i_rest)
            results = []
            qa_pairs = self.get_qa_pairs(section)
            index = len(dialogues)
            for qa_pair in qa_pairs:
                q = {
                    "index": index,
                    "vector": qa_pair['question_vector']
                }
                data = self.prepare_data(faiss, sr_sql, q['index'], q['vector'])
                sr_results = self.propagate_stimuli(faiss, sr_sql, data, cos_th, v_th, stimulus_th, tau_init, tau_scale, time_scale, bond_scale, v_rest, i_rest)
                sr_memories = sr_sql.get_all_data()
                for memory in sr_memories:
                    if memory['id'] == q['index']:
                        continue
                    if memory['id'] in sr_results:
                        continue
                    sr_results[memory['id']] = memory['v']
                result = {}
                result['index'] = str(section) + '_' + str(q['index'])
                result['sr'] = sr_results
                results.append(result)
                index += 1
                a = {
                    "index": index,
                    "vector": qa_pair['answer_vector']
                }
                data = self.prepare_data(faiss, sr_sql, a['index'], a['vector'])
                sr_results = self.propagate_stimuli(faiss, sr_sql, data, cos_th, v_th, stimulus_th, tau_init, tau_scale, time_scale, bond_scale, v_rest, i_rest)
                index += 1
            return results
        all_results = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_section, section) for section in dataset]
            with tqdm(total=section_count) as pbar:
                for future in as_completed(futures):
                    pbar.update(1)
                    result = future.result()
                    all_results.extend(result)
        save_path = self.analyze_dir + f'/{param}/{sr_params[param]}.json'
        if not os.path.exists(self.analyze_dir + f'/{param}'):
            os.makedirs(self.analyze_dir + f'/{param}')
        with open(save_path, 'w') as f:
            json.dump(all_results, f, indent=4)
        self.clear_directory()

    def get_dialogues(self, section):
        conn = sqlite3.connect(self.dialogue_db_path)
        cursor = conn.cursor()
        query = """
        SELECT dialogue_index, text, vector
        FROM dialogue_vectors
        WHERE section = ?
        ORDER BY dialogue_index
        """
        cursor.execute(query, (section,))
        results = cursor.fetchall()
        dialogues = []
        for index, text, vector in results:
            vector_data = json.loads(vector)
            dialogues.append({
                "index": index,
                "text": text,
                "vector": vector_data
            })
        dialogues = sorted(dialogues, key=lambda x: x['index'])
        conn.close()
        return dialogues

    def get_qa_pairs(self, section):
        conn = sqlite3.connect(self.qa_db_path)
        cursor = conn.cursor()
        query = """
        SELECT qa_index, text, recall, vector
        FROM qa_vectors
        WHERE section = ?
        ORDER BY qa_index
        """
        cursor.execute(query, (section,))
        results = cursor.fetchall()
        qa_pairs = []
        for index, text, recall, vector in results:
            vector_data = json.loads(vector)
            text_data = json.loads(text)
            qa_pairs.append({
                "index": index,
                "question": text_data[0],
                "answer": text_data[1],
                "recall": recall,
                "question_vector": vector_data[0],
                "answer_vector": vector_data[1]
            })
        qa_pairs = sorted(qa_pairs, key=lambda x: x['index'])
        conn.close()
        return qa_pairs