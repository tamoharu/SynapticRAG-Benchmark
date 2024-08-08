import sys
sys.path.append('../..')
import json
import asyncio
import sqlite3
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from utils.request import Request
from utils.log import Log


lang = 'en'


log = Log('info')


if lang == 'en':
    dataset_path = './test/dataset_EN.json'
    db_path = './prepared/vectors_en.db'
elif lang == 'ja':
    dataset_path = './test/dataset_JA.json'
    db_path = './prepared/vectors_ja.db'
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

@retry(
    retry=retry_if_exception_type((httpx.ConnectError, httpx.HTTPStatusError)),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    before_sleep=lambda retry_state: log.error(f"Retrying in {retry_state.next_action.sleep} seconds...") # type: ignore
)
async def vectorize_text(text):
    request = Request()
    try:
        result = await request.async_embed(text)
        if result is None or not result:
            raise ValueError("Received empty result from embedding API")
        return result
    except httpx.ConnectError as e:
        log.error(f"Connection error occurred: {e}")
        raise
    except httpx.HTTPStatusError as e:
        log.error(f"HTTP error occurred: {e}")
        raise
    except Exception as e:
        log.error(f"An unexpected error occurred: {e}")
        raise

class Vector():
    def __init__(self):
        self.init_db()

    def init_db(self):
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS vectors
                     (section INTEGER, text_index INTEGER, text TEXT, recall TEXT, vector TEXT,
                      PRIMARY KEY (section, text_index))''')
        conn.commit()
        conn.close()

    async def prepare_dialogue_vectors(self):
        dataset = read_json_data(dataset_path)
        all_texts = []
        text_info = []

        for section, data in enumerate(dataset):
            past = data['past']
            present = data['present']
            recalls = data['recalls']
            section_texts = []
            for content in past:
                for text in content.values():
                    section_texts.append(text)
                    all_texts.append(text)
            for content in present:
                for text in content.values():
                    section_texts.append(text)
                    all_texts.append(text)

            recall_map = {}
            for recall in recalls:
                trigger_text = recall['trigger']
                related_memories = recall['related_memories']
                recall_map[trigger_text] = related_memories

            def get_dialogue_indices(dialogues, text):
                indices = []
                for memory in recall_map[text]:
                    for i, dialogue in enumerate(dialogues):
                        if dialogue == memory:
                            indices.append(i)
                if len(indices) == 0:
                    log.error(f"Failed to find memory indices for text: {text}")
                return indices

            for index, content in enumerate(past):
                for text in content.values():
                    memory_indices = get_dialogue_indices(section_texts, text) if text in recall_map else []
                    text_info.append([section, index, text, memory_indices])

            for index, content in enumerate(present):
                for text in content.values():
                    memory_indices = get_dialogue_indices(section_texts, text) if text in recall_map else []
                    text_info.append([section, len(past) + index, text, memory_indices])

        log.info(f"Total texts to vectorize: {len(all_texts)}")
        chunk_size = 100
        all_vectors = []
        for i in range(0, len(all_texts), chunk_size):
            chunk = all_texts[i:i+chunk_size]
            try:
                vectors = await asyncio.gather(*[vectorize_text(text) for text in chunk])
                all_vectors.extend(vectors)
                log.info(f"Vectorized texts {i+1}-{i+len(vectors)}/{len(all_texts)}")
            except Exception as e:
                log.error(f"Error vectorizing texts {i+1}-{i+len(chunk)}/{len(all_texts)}: {e}")
                all_vectors.extend([None] * len(chunk))
        log.info(f"Total vectors generated: {len(all_vectors)}")
        log.info("Saving to database...")
        await self.save_to_db(text_info, all_vectors)
        log.info("Finished saving to database.")

    async def save_to_db(self, text_info, vectors):
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        for i, ((section, index, text, recall), vector) in enumerate(zip(text_info, vectors)):
            if vector is not None:
                try:
                    c.execute('''INSERT OR REPLACE INTO vectors 
                                 (section, text_index, text, recall, vector) 
                                 VALUES (?, ?, ?, ?, ?)''',
                              (section, index, text, json.dumps(recall), json.dumps(vector)))
                    if i % 100 == 0:
                        log.info(f"Saved vector {i+1}/{len(vectors)} to database")
                except Exception as e:
                    log.error(f"Failed to save vector {i+1}/{len(vectors)} to database: {e}")
            else:
                log.error(f"Skipped saving vector {i+1}/{len(vectors)} (None value)")

        conn.commit()
        conn.close()