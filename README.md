# How to run

## Conda Environments

```bash
conda create --name NavGPT python=3.9
conda activate NavGPT
pip install -r requirements.txt
```

## Data Preparation

Follow the instructions in [NavGPT's repository](https://github.com/gengzezhou/navgpt#-data-preparation)

## Install ollama

Install `modelscope`.
```bash
pip install modelscope
```

Download `ollama`.
```bash
modelscope download --model=modelscope/ollama-linux --local_dir ./ollama-linux --revision v0.5.7
```

Check the installation.
```bash
ollama -v
```

```bash
cd ollama-linux
chmod +x ./ollama-modelscope-install.sh
```

Then run
```bash
./ollama-modelscope-install.sh
```

### Start Ollama server

**Ollama server should be running when using *any* ollama commands.**
```bash
ollama serve
```
Add `CUDA_VISIBLE_DEVICES` if needed. (Highly recommanded)

### Pull Ollama models

```bash
ollama pull [model-name]
```

For example, `ollama pull deepseek-r1:7b`. More models can be found at [Ollama models](https://ollama.com/search).

## Run experiments

```bash
cd NavGPT/nav_src
```

```bash
CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch NavGPT.py --llm_model_name ollama-deepseek --val_env_name R2R_val_unseen_instr --output_dir ../datasets/R2R/exprs/ollama-deepseek-test-1 --iters 10
```
