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
        self.conv_log_path_1 = os.path.abspath(os.path.join(os.path.dirname(__file__), './log/conv_log_1.txt'))
        self.conv_log_path_2 = os.path.abspath(os.path.join(os.path.dirname(__file__), './log/conv_log_2.txt'))
        self.conv_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../process_files'))
        self.count = 0
        if lang == 'en':
            self.dialogue_path = os.path.abspath(os.path.join(os.path.dirname(__file__), './dataset/prepared/filtered_dialogue.json'))
            self.qa_path = os.path.abspath(os.path.join(os.path.dirname(__file__), './dataset/prepared/filtered_qa.json'))
            self.dialogue_db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), './dataset/prepared/dialogue.db'))
            self.qa_db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), './dataset/prepared/qa.db'))
        elif lang == 'zh':
            self.dialogue_path = os.path.abspath(os.path.join(os.path.dirname(__file__), './dataset/prepared/filtered_dialogue_zh.json'))
            self.qa_path = os.path.abspath(os.path.join(os.path.dirname(__file__), './dataset/prepared/filtered_qa_zh.json'))
            self.dialogue_db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), './dataset/prepared/dialogue_zh.db'))
            self.qa_db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), './dataset/prepared/qa_zh.db'))
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
            section_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in section)
            faiss = Faiss(dimension=3072, index_path=f'section_{section_name}.index')
            sr_sql = SQL(f'section_{section_name}.db', "SynapticRAG")
            ma_sql = SQL(f'section_{section_name}.db', "MyAent")
            ma_ex_sql = SQL(f'section_{section_name}.db', "MyAgent_ex")
            mb_sql = SQL(f'section_{section_name}.db', "MemoryBank")
            mb_ex_sql = SQL(f'section_{section_name}.db', "MemoryBank_ex")
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
                self.prepare_dialogue(dialogue, faiss, sr_sql, ma_sql, ma_ex_sql, mb_sql, mb_ex_sql, sr_params, ma_params, ma_ex_params, mb_ex_params)

            qa_pairs = self.get_qa_pairs(section)
            index = len(dialogues)
            for qa_pair in qa_pairs:
                local_tasks += 1
                q = {
                    "index": index,
                    "vector": qa_pair['question_vector']
                }
                sr_results, ma_results, ma_ex_results, mb_results, mb_ex_results = self.prepare_dialogue(q, faiss, sr_sql, ma_sql, ma_ex_sql, mb_sql, mb_ex_sql, sr_params, ma_params, ma_ex_params, mb_ex_params, measure_time=measure_time)
                recall_indices = [qa_pair['recall']]
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

                import time
                start_time = time.time()
                rag_search = faiss.search_embeddings(query_vector=q['vector'], exclude_ids=[index])
                if measure_time:
                    print(f"RAG search time: {time.time() - start_time}")
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

                self.result_log(self.conv_log_path_1, section, index, recall_indices, sr_results, rag_results_1, ma_results_1, ma_ex_results_1, mb_results_1, mb_ex_results_1)
                self.result_log(self.conv_log_path_2, section, index, recall_indices, sr_results, rag_results_2, ma_results_2, ma_ex_results_2, mb_results_2, mb_ex_results_2)
                index += 1
                a = {
                    "index": index,
                    "vector": qa_pair['answer_vector']
                }
                self.prepare_dialogue(a, faiss, sr_sql, ma_sql, ma_ex_sql, mb_sql, mb_ex_sql, sr_params, ma_params, ma_ex_params, mb_ex_params)
                index += 1

            return local_sr, local_ma_1, local_ma_2, local_ma_ex_1, local_ma_ex_2, local_mb_1, local_mb_2, local_mb_ex_1, local_mb_ex_2, local_rag_1, local_rag_2, local_tasks

        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = [executor.submit(process_section, section) for section in dataset]

            with tqdm(total=section_count) as pbar:
                for future in as_completed(futures):
                    results = future.result()
                    sr += results[0]
                    ma_1 += results[1]
                    ma_2 += results[2]
                    ma_ex_1 += results[3]
                    ma_ex_2 += results[4]
                    mb_1 += results[5]
                    mb_2 += results[6]
                    mb_ex_1 += results[7]
                    mb_ex_2 += results[8]
                    rag_1 += results[9]
                    rag_2 += results[10]
                    tasks += results[11]
                    pbar.update(1)

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

        print(f'section_count: {section_count}')
        print(f'tasks: {tasks}')
        print(f'\nEvaluation 1')
        print(f"Model_ratio: {model_ratio}%, RAG_ratio: {rag_ratio_1}%, MyAgent_ratio: {ma_ratio_1}%, MyAgent_ex_ratio: {ma_ex_ratio_1}%, MemoryBank_ratio: {mb_ratio_1}%, MemoryBank_ex_ratio: {mb_ex_ratio_1}%")
        print(f"Model_count: {sr}, RAG_count: {rag_1}, MyAgent_count: {ma_1}, MyAgent_ex_count: {ma_ex_1}, MemoryBank_count: {mb_1}, MemoryBank_ex_count: {mb_ex_1}")
        print(f'\nEvaluation 2')
        print(f"Model_ratio: {model_ratio}%, RAG_ratio: {rag_ratio_2}%, MyAgent_ratio: {ma_ratio_2}%, MyAgent_ex_ratio: {ma_ex_ratio_2}%, MemoryBank_ratio: {mb_ratio_2}%, MemoryBank_ex_ratio: {mb_ex_ratio_2}%")
        print(f"Model_count: {sr}, RAG_count: {rag_2}, MyAgent_count: {ma_2}, MyAgent_ex_count: {ma_ex_2}, MemoryBank_count: {mb_2}, MemoryBank_ex_count: {mb_ex_2}")
        self.clear_directory()
        return {'sr': sr, 'rag': rag_1, 'ma': ma_1, 'ma_ex': ma_ex_1, 'mb': mb_1, 'mb_ex': mb_ex_1}

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