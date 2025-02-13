import sys
sys.path.append('..')
import os
import json
import sqlite3
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.sql import SQL
from utils.faiss import Faiss
from utils.eval import Eval

measure_time = False

class EvalProcess(Eval):
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
        elif lang == 'zh':
            self.dialogue_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../datasets/PerLTQA/dataset/prepared/filtered_dialogue_zh.json'))
            self.qa_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../datasets/PerLTQA/dataset/prepared/filtered_qa_zh.json'))
            self.dialogue_db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../datasets/PerLTQA/dataset/prepared/dialogue_zh.db'))
            self.qa_db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../datasets/PerLTQA/dataset/prepared/qa_zh.db'))
            self.result_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../results/PerLTQA (CN)/'))
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
                self.prepare_dialogue(dialogue, faiss, sr_sql, ma_sql, ma_ex_sql, mb_sql, mb_ex_sql, sr_params, ma_params, ma_ex_params, mb_ex_params)
            qa_pairs = self.get_qa_pairs(section)
            index = len(dialogues)
            for qa_pair in qa_pairs:
                q = {
                    "index": index,
                    "vector": qa_pair['question_vector']
                }
                sr_results, ma_results, ma_ex_results, mb_results, mb_ex_results = self.prepare_dialogue(q, faiss, sr_sql, ma_sql, ma_ex_sql, mb_sql, mb_ex_sql, sr_params, ma_params, ma_ex_params, mb_ex_params)
                rag_search = faiss.search_embeddings(query_vector=q['vector'], exclude_ids=[index])
                rag_results = {}
                for id, distance in zip(rag_search[0], rag_search[1]):
                    rag_results[id] = distance
                sr_memories = sr_sql.get_all_data()
                for memory in sr_memories:
                    if memory['id'] == qa_pair['index']:
                        continue
                    if memory['id'] in sr_results:
                        continue
                    sr_results[memory['id']] = memory['v']
                result = {}
                result['index'] = str(section) + '_' + str(q['index'])
                result['recall'] = [qa_pair['recall']]
                result['sr'] = sr_results
                result['ma'] = ma_results
                result['ma_ex'] = ma_ex_results
                result['mb'] = mb_results
                result['mb_ex'] = mb_ex_results
                result['rag'] = rag_results
                results.append(result)
                index += 1
                a = {
                    "index": index,
                    "vector": qa_pair['answer_vector']
                }
                self.prepare_dialogue(a, faiss, sr_sql, ma_sql, ma_ex_sql, mb_sql, mb_ex_sql, sr_params, ma_params, ma_ex_params, mb_ex_params)
                index += 1
            return results

        all_results = []
        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = [executor.submit(process_section, section) for section in dataset]
            with tqdm(total=section_count) as pbar:
                for future in as_completed(futures):
                    pbar.update(1)
                    result = future.result()
                    all_results.extend(result)
        with open(self.result_dir + '/memories.json', 'w') as f:
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