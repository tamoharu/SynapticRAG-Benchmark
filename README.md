# ⚠️ Note on SynapticRAG

**Important:**
This repository corresponds to ACL 2025 paper **by Yuki Hou, Haruki Tamoto et al.** titled **"[SynapticRAG: Enhancing Temporal Memory Retrieval in Large Language Models through Synaptic Mechanisms](https://arxiv.org/abs/2410.13553)"** (arXiv [v1](https://arxiv.org/abs/2410.13553) submitted on Oct 17, 2024)

**This work is completely distinct from** another paper with a similar title: **"SynapticRAG: Integrating LightRAG and MemoRAG for Enhanced Retrieval-Augmented Generation with Memory Recall and Knowledge Graphs"** by Abhinav Agarwal et al., which proposes "a hybrid RAG system combining MemoRAG's with LightRAG's powered by PyTorch." [[Article](https://medium.com/stanford-cs224w/synapticrag-integrating-lightrag-and-memorag-for-enhanced-retrieval-augmented-generation-with-3935d0eb45d0)] [[GitHub](https://github.com/abhinav30219/synapticrag)] (Published Dec 13, 2024)

We respectfully clarify that the above-mentioned paper describes a completely different model from ours, proposing distinctly different methods and objectives.

# ✅ SynapticRAG Benchmark

## Setup

**Installation**

Run the following command:

```
python install.py
```

## Quick Start

Main Results

```
python eval.py --dataset SMRCs --lang en
```

Ablation Study

```
python ablation.py --dataset SMRCs --lang en
```

Parameter Analysis

```
python analyze.py --dataset SMRCs --lang en
```

You can see the results in the results directory.

## Options

Select the dataset and language:

- PerLTQA:

```
--dataset PerLTQA --lang en
--dataset PerLTQA --lang zh
```

- SMRCs:

```
--dataset SMRCs --lang en
--dataset SMRCs --lang ja
```
