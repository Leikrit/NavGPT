# 🤖 How to run

## 🔧 Conda Environments

```bash
conda create --name NavGPT python=3.9
conda activate NavGPT
pip install -r requirements.txt
```

## 📦 Data Preparation

Follow the instructions in [NavGPT's repository](https://github.com/gengzezhou/navgpt#-data-preparation)

## 🦙 Install ollama

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

## 📟 Run experiments

```bash
cd NavGPT/nav_src
```

```bash
CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch NavGPT.py --llm_model_name ollama-deepseek --val_env_name R2R_val_unseen_instr --output_dir ../datasets/R2R/exprs/ollama-deepseek-test-1 --iters 10
```

# 👁️‍🗨️ Visualization

Visualization can be done following the instructions below.

## 📃 Running exprs

Firstly, the visualization module is based on **experiment logs**, which means you need to run some experiments first. You can follow the guidance above to run the experiments.

## ⬆️ Extract trajectories

Run `visualization.py` to extract the trajectories from the experiment logs, and transfer them into coordinates for habitat simulator.
> Currently, `visualization.py` is not yet implemented, please wait for updates coming soon.

After extracting trajectories, all trajectories together with the scene ID will be saved into a JSON file. You can rename it in `visualization.py`.

## 📍🗺️ Visualize trajectories

Run `draw_on_map.py` to visualize the experiment logs one at a time. Then, you will get a map with trajectory drawn, an image of the agent's view and an image of panorama view.

