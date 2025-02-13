# SynapticRAG Benchmark

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

you can see the each results in `results` directory.

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