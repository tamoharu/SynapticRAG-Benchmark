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
        self.conv_log_path_1 = os.path.abspath(os.path.join(os.path.dirname(__file__), '../datasets/SMRCs/log/conv_log_1.txt'))
        self.conv_log_path_2 = os.path.abspath(os.path.join(os.path.dirname(__file__), '../datasets/SMRCslog/conv_log_2.txt'))
        self.conv_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../process_files'))
        self.count = 0
        if lang == 'en':
            self.dialogue_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dataset/SMRCs/dataset_EN.json'))
            self.db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../datasets/SMRCs/dataset/prepared/vectors_en.db'))
            self.result_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../results/SMRCs (EN)/'))
            self.analyze_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../analyze/SMRCs (EN)/'))
        elif lang == 'ja':
            self.dialogue_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dataset/SMRCs/dataset_JA.json'))
            self.db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../datasets/SMRCs/dataset/prepared/vectors_ja.db'))
            self.result_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../results/SMRCs (JA)/'))
            self.analyze_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../analyze/SMRCs (JA)/'))
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
            results = []
            for dialogue in dialogues:
                recall_indices = dialogue['recall']
                data = self.prepare_data(faiss, sr_sql, dialogue['index'], dialogue['vector'])
                sr_results = self.propagate_stimuli(faiss, sr_sql, data, cos_th, v_th, stimulus_th, tau_init, tau_scale, time_scale, bond_scale, v_rest, i_rest)
                if len(recall_indices) > 0:
                    sr_memories = sr_sql.get_all_data()
                    for memory in sr_memories:
                        if memory['id'] == dialogue['index']:
                            continue
                        if memory['id'] in sr_results:
                            continue
                        sr_results[memory['id']] = memory['v']
                    result = {}
                    result['index'] = str(section) + '_' + str(dialogue['index'])
                    result['sr'] = sr_results
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
        save_path = self.analyze_dir + f'/{param}/{sr_params[param]}.json'
        if not os.path.exists(self.analyze_dir + f'/{param}'):
            os.makedirs(self.analyze_dir + f'/{param}')
        with open(save_path, 'w') as f:
            json.dump(all_results, f, indent=4)
        self.clear_directory()

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