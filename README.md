# SynapticRAG Benchmark

## Setup

**Installation**

Run the following command:

```
python install.py
```

## Available Model and Dataset Names

### Model Names

- SynapticRAG:

```
--model sr
```

- MemoryBank:

```
--model mb
```

- MemoryBank (Extended)

```
--model mb_ex
```

- MyAgent:

```
--model ma
```

- MyAgent (Extended):

```
--model ma_ex
```

### Dataset Names

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

## Optimization

Select the model name in the root directory. 

Example:

```
python optimize.py --model sr
```

## Evaluation

Select the dataset. 

Example:

```
python run.py --dataset SMRCs --lang en
```