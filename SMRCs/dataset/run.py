import asyncio
from dataset import Vector


async def run():
    vec = Vector()
    await vec.prepare_dialogue_vectors()


if __name__ == '__main__':
    asyncio.run(run())