import asyncio
from dataset import lang, JSON, DialogueVector, QAVector, filter


async def prepare_data(language):
    lang = language
    json = JSON()
    json.prepare_dialogues()
    json.prepare_qa()
    filter()
    diavec = DialogueVector()
    await diavec.prepare_dialogue_vectors()
    qavec = QAVector()
    await qavec.process()


async def run():
    await prepare_data('en')
    await prepare_data('zh')


if __name__ == '__main__':
    asyncio.run(run())