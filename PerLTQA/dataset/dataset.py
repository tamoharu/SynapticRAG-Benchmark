import sys
sys.path.append('../..')
import json
import asyncio
import sqlite3
import httpx
import numpy
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from utils.request import Request


lang = 'en'


if lang == 'en':
    dialogue_path = './prepared/dialogue.json'
    qa_path = './prepared/qa.json'
    filtered_dialogue_path = './prepared/filtered_dialogue.json'
    filtered_qa_path = './prepared/filtered_qa.json'
    dialogue_db_path = './prepared/dialogue.db'
    qa_db_path = './prepared/qa.db'
elif lang == 'zh':
    dialogue_path = './prepared/dialogue_zh.json'
    qa_path = './prepared/qa_zh.json'
    filtered_dialogue_path = './prepared/filtered_dialogue_zh.json'
    filtered_qa_path = './prepared/filtered_qa_zh.json'
    dialogue_db_path = './prepared/dialogue_zh.db'
    qa_db_path = './prepared/qa_zh.db'
else:
    raise ValueError(f"Unsupported language: {lang}")

def read_json_data(path):
    with open(path, 'r', encoding="utf-8") as f:
        content = f.read()
        dataset = json.loads(content)
    return dataset

def save_to_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


class JSON:
    def __init__(self):
        if lang == 'en':
            self.dialogue_raw_path = './PerLTQA/perltmem_en.json'
            self.qa_raw_path = './PerLTQA/perltqa_en.json'
        elif lang == 'zh':
            self.dialogue_raw_path = './PerLTQA/perltmem.json'
            self.qa_raw_path = './PerLTQA/perltqa.json'
    
    def remove_prefix(self, text):
        if isinstance(text, str):
            colon_index = text.find(":")
            if colon_index != -1:
                return text[colon_index + 2:].strip()
        return text
    
    def prepare_dialogues(self):
        dataset = read_json_data(self.dialogue_raw_path)
        all_dialogues = {}
        for content in dataset:
            data = content["dialogues"]
            for section, value in data.items():
                texts = []
                contents = value["contents"]
                if len(contents) == 0:
                    continue
                for _, text in contents.items():
                    for t in text:
                        if isinstance(t, list):
                            for tt in t:
                                tt = self.remove_prefix(tt)
                                texts.append(tt)
                        else:
                            t = self.remove_prefix(t)
                            texts.append(t)
                all_dialogues[section] = texts
        save_to_json(all_dialogues, dialogue_path)
    
    def prepare_qa(self):
        dataset = read_json_data(self.qa_raw_path)
        all_qa = {}
        for content in dataset:
            character_name = list(content.keys())[0]
            d = content[character_name]
            data = d["dialogues"]
            if lang == 'en':
                for datum in data:
                    section = list(datum.keys())[0]
                    value = datum[section]
                    qas = []
                    for obj in value:
                        qa = {}
                        qa["Question"] = obj["Question"]
                        qa["Answer"] = obj["Answer"]
                        qas.append(qa)
                    all_qa[section] = qas
                save_to_json(all_qa, qa_path)
            elif lang == 'zh':
                sections = list(data.keys())
                for section in sections:
                    value = data[section]
                    qas = []
                    for obj in value:
                        qa = {}
                        qa["Question"] = obj["Question"]
                        qa["Answer"] = obj["Answer"]
                        qas.append(qa)
                    all_qa[section] = qas
                    save_to_json(all_qa, qa_path)

class Vector:
    @retry(
        retry=retry_if_exception_type((httpx.ConnectError, httpx.HTTPStatusError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=lambda retry_state: print(f"Retrying in {retry_state.next_action.sleep} seconds...") # type: ignore
    )
    async def vectorize_text(self, text):
        request = Request()
        try:
            result = await request.async_embed(text)
            if result is None or not result:
                raise ValueError("Received empty result from embedding API")
            return result
        except httpx.ConnectError as e:
            print(f"Connection error occurred: {e}")
            raise
        except httpx.HTTPStatusError as e:
            print(f"HTTP error occurred: {e}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise
    

class DialogueVector(Vector):
    def __init__(self):
        super().__init__()
        self.init_db()

    def init_db(self):
        conn = sqlite3.connect(dialogue_db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS dialogue_vectors
                     (section TEXT, dialogue_index INTEGER, text TEXT, vector TEXT,
                      PRIMARY KEY (section, dialogue_index))''')
        conn.commit()
        conn.close()

    async def prepare_dialogue_vectors(self):
        dataset = read_json_data(filtered_dialogue_path)
        all_texts = []
        text_info = []
        for section, dialogue in dataset.items():
            for k, text in enumerate(dialogue):
                all_texts.append(text)
                text_info.append((section, k, text))
        chunk_size = 1000
        all_vectors = []
        for i in range(0, len(all_texts), chunk_size):
            chunk = all_texts[i:i+chunk_size]
            try:
                vectors = await asyncio.gather(*[self.vectorize_text(text) for text in chunk])
                all_vectors.extend(vectors)
                print(f"Vectorized texts {i+1}-{i+len(vectors)}/{len(all_texts)}")
            except Exception as e:
                print(f"Error vectorizing texts {i+1}-{i+len(chunk)}/{len(all_texts)}: {e}")
                all_vectors.extend([None] * len(chunk))
        print("Saving to database...")
        await self.save_to_db(text_info, all_vectors)
        print("Finished saving to database.")

    async def save_to_db(self, text_info, vectors):
        conn = sqlite3.connect(dialogue_db_path)
        c = conn.cursor()
        for i, ((section, dialogue_index, text), vector) in enumerate(zip(text_info, vectors)):
            if vector is not None:
                try:
                    c.execute('''INSERT OR REPLACE INTO dialogue_vectors 
                                 (section, dialogue_index, text, vector) 
                                 VALUES (?, ?, ?, ?)''',
                              (section, dialogue_index, text, json.dumps(vector)))
                    if i % 100 == 0:
                        print(f"Saved vector {i+1}/{len(vectors)} to database")
                except Exception as e:
                    print(f"Failed to save vector {i+1}/{len(vectors)} to database: {e}")
            else:
                print(f"Skipped saving vector {i+1}/{len(vectors)} (None value)")
        
        conn.commit()
        conn.close()


class QAVector(Vector):
    def __init__(self):
        super().__init__()
        self.init_db()
    
    def init_db(self):
        conn = sqlite3.connect(qa_db_path)
        c = conn.cursor()
        c.execute('DROP TABLE IF EXISTS qa_vectors')
        c.execute('''CREATE TABLE IF NOT EXISTS qa_vectors
                     (section TEXT, qa_index INTEGER, text TEXT, recall INTEGER, vector TEXT,
                      PRIMARY KEY (section, qa_index))''')
        conn.commit()
        conn.close()

    async def process(self):
        dataset = read_json_data(filtered_dialogue_path)
        all_qa = await self.prepare_qa_vectors()
        for section, _ in dataset.items():
            dataset = read_json_data(filtered_qa_path)
            qa_pairs = all_qa[section]
            for qa_index, qa_pair in qa_pairs.items():
                answer_vector = qa_pair["answer_vector"]
                dialogues = self.get_dialogues(section)
                dialogue_vectors = [dialogue["vector"] for dialogue in dialogues]
                most_similar_index, max_similarity = self.find_most_similar_vector(answer_vector, dialogue_vectors)
                if max_similarity >= 0.5:
                    question_text = qa_pair["question"]
                    answer_text = qa_pair["answer"]
                    await self.save_to_db(section, qa_pair["index"], [question_text, answer_text], most_similar_index, [qa_pair["question_vector"], qa_pair["answer_vector"]])

    async def prepare_qa_vectors(self):
        dataset = read_json_data(filtered_qa_path)
        all_texts = []
        text_info = []
        for section, qas in dataset.items():
            for qa_idx, qa in enumerate(qas):
                question = qa.get("Question", "")
                answer = qa.get("Answer", "")
                if question:
                    all_texts.append(question)
                    text_info.append((section, "question", qa_idx, question))
                if answer:
                    all_texts.append(answer)
                    text_info.append((section, "answer", qa_idx, answer))
            print(f"    Added Q&A for section {section}")
        print("Starting vectorization...")
        chunk_size = 100
        all_vectors = []
        for i in range(0, len(all_texts), chunk_size):
            chunk = all_texts[i:i+chunk_size]
            try:
                vectors = await asyncio.gather(*[self.vectorize_text(text) for text in chunk])
                all_vectors.extend(vectors)
                print(f"Vectorized texts {i+1}-{i+len(vectors)}/{len(all_texts)}")
            except Exception as e:
                print(f"Error vectorizing texts {i+1}-{i+len(chunk)}/{len(all_texts)}: {e}")
                all_vectors.extend([None] * len(chunk))
        all_qa = {}
        for i, ((section, qa_type, qa_index, text), vector) in enumerate(zip(text_info, all_vectors)):
            if vector is not None:
                if section not in all_qa:
                    all_qa[section] = {}
                if qa_type == "question":
                    all_qa[section][qa_index] = {
                        "index": qa_index,
                        "question": text,
                        "question_vector": vector
                    }
                elif qa_type == "answer":
                    all_qa[section][qa_index]["answer"] = text
                    all_qa[section][qa_index]["answer_vector"] = vector
        return all_qa

    def get_dialogues(self, section):
        conn = sqlite3.connect(dialogue_db_path)
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
    
    async def save_to_db(self, section, qa_index, text, recall, vector):
        conn = sqlite3.connect(qa_db_path)
        c = conn.cursor()
        c.execute('''INSERT OR REPLACE INTO qa_vectors 
                    (section, qa_index, text, recall, vector) 
                    VALUES (?, ?, ?, ?, ?)''',
                (section, qa_index, json.dumps(text), recall, json.dumps(vector)))
        conn.commit()
        conn.close()
    
    def find_most_similar_vector(self, query_vector, vector_list):
        max_similarity = -1
        most_similar_index = -1
        for i, vector in enumerate(vector_list):
            similarity = self.cosine_similarity(query_vector, vector)
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_index = i
        return most_similar_index, max_similarity
    
    def cosine_similarity(self, v1, v2):
        v1, v2 = numpy.array(v1), numpy.array(v2)
        dot_product = numpy.dot(v1, v2)
        norm_v1 = numpy.sqrt(numpy.sum(v1**2))
        norm_v2 = numpy.sqrt(numpy.sum(v2**2))
        similarity = dot_product / (norm_v1 * norm_v2)
        return similarity


def filter():
    dialogue = read_json_data(dialogue_path)
    qa = read_json_data(qa_path)
    common_sections = set(dialogue.keys()).intersection(set(qa.keys()))
    filtered_dialogue = {key: value for key, value in dialogue.items() if key in common_sections}
    filtered_qa = {key: value for key, value in qa.items() if key in common_sections}
    save_to_json(filtered_dialogue, filtered_dialogue_path)
    save_to_json(filtered_qa, filtered_qa_path)