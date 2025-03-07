<div align="center">

<h1>🎇NavGPT: Explicit Reasoning in Vision-and-Language Navigation with Large Language Models</h1>

<div>
    Originally from <a href='https://github.com/GengzeZhou' target='_blank'>Gengze Zhou</a> et al. </br>
    Modified by
    <a href='https://github.com/Leikrit' target='_blank'>Jinyi Li</a><sup>🌟</sup>
</div>
<sup>🌟</sup>School of Computer Science and Engineering, South China University of Technology

</div>

# 🤖 How to run

## 🔧 Conda Environments

```bash
conda create --name NavGPT python=3.9
conda activate NavGPT
pip install -r requirements.txt
```

## 📦 Data Preparation

Follow the instructions in [NavGPT's repository](https://github.com/gengzezhou/navgpt#-data-preparation).

Download R2R data from [Dropbox](https://www.dropbox.com/sh/i8ng3iq5kpa68nu/AAB53bvCFY_ihYx1mkLlOB-ea?dl=1). Put the data in `datasets` directory.

Related data preprocessing code can be found in `nav_src/scripts`.

## 🦙 Install ollama

### 🇨🇳 For Chinese users

Install `modelscope`.
```bash
pip install modelscope
```

Download `ollama`.
```bash
modelscope download --model=modelscope/ollama-linux --local_dir ./ollama-linux --revision v0.5.7
```

```bash
cd ollama-linux
chmod +x ./ollama-modelscope-install.sh
```

Then run
```bash
./ollama-modelscope-install.sh
```

Check the installation.
```bash
ollama -v
```

### 🌐 For other regions

Just simply follow the guidance on [Ollama](https://ollama.com/download/linux).

---
---

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

## 🏠 Installing Habitat

Installing Habitat v0.3.3 is strongly recommended. 

Following the instructions in <a href="https://github.com/facebookresearch/habitat-sim"> Habitat official </a>. 

Remember to **install Habitat lab** as well, or you might encounter `No module named habitat` error.

> Why v0.3.3? Because with such version, the FOV of the cameras can be modified in order to perfectly fit the setting in NavGPT's original paper.

### With lower version

For example, if your Habitat version is v0.1.7, that you should change the `make_new_cfg()` function into `make_cfg()` in `draw_on_map.py`.

Also, you should modify the `sim_settings` variable. Delete the key `"scene_dataset"` as it is no longer needed for lower version.

## 📦 Data preparation

Prepare the data following <a href="https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#matterport3d-mp3d-dataset"> Habitat official </a>. You only need to download `mp3d` dataset.

If you don't have `scene_dataset_config.json` file in your dataset folders, you can also download it from the link above.

> This file is for higher versions, ignore it if you are using lower version

> By the way, I am sorry that I don't know which version is high or low. You can check whether your version has the class `CameraSensorSpec`. If do, then you are high, otherwise you are low.

## 📃 Running exprs

Firstly, the visualization module is based on **experiment logs**, which means you need to run some experiments first. You can follow the guidance above to run the experiments.

## ⬆️ Extract trajectories

Run `visualization.py` to extract the trajectories from the experiment logs, and transfer them into coordinates for habitat simulator.
> Remember to change the file path in `visualization.py`.

After extracting trajectories, all trajectories together with the scene ID will be saved into a JSON file. You can rename it in `visualization.py`.

## 📍🗺️ Visualize trajectories

Run `draw_on_map.py` to visualize the experiment logs one at a time. Then, you will get a map with trajectory drawn, an image of the agent's view and an image of panorama view.
> Remember to change the file path in `draw_on_map.py`.

Now that with the newest version, the visualization results will be stored in the `imgs` folder in the same directory with `draw_on_map.py`. The `imgs` folder will look like:

```bash
|- [scene ID]_1
    |- [scene ID]_1_0_original.jpg
    |- [scene ID]_1_0_panorama.jpg
    |- ...
    |- [scene ID]_1_n_original.jpg
    |- [scene ID]_1_n_panorama.jpg
    |- {scene ID}_1_map.jpg
|- ...
```

Where 'original' means the original image that the agent sees, 'panorama' means the original panorama, which is basically a combination of 8 images covering 360 degrees.
