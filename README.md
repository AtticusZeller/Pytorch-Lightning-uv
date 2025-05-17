<div align="center">

# 🧪 Pytorch Lightning UV Template

A modern Deep Learning experiment template with PyTorch Lightning

[![python](https://img.shields.io/badge/-Python_3.10_%7C_3.11_%7C_3.12-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-310/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.5+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.5+-792ee5?logo=pytorchlightning&logoColor=white)](https://lightning.ai/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/ashleve/lightning-hydra-template#license)
[![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/AtticusZeller/Pytorch-Lightning-uv/pulls)

Click on [<kbd>Use this template</kbd>](https://github.com/AtticusZeller/Pytorch-Lightning-uv/generate) to initialize new repository.

</div>

## ✨ Features

* 🚀 **UV Environment Management** - Fast and efficient dependency management
* 🎯 **Typer CLI** - Modern command line interface
* ⚙️ **YAML Config** - Flexible experiment configuration
* 🔋 **Lightning Components**
  + DataModule for clean data handling
  + Model with built-in training logic
  + Trainer with all the bells and whistles
* 📊 **Weights & Biases Integration**
  + Experiment tracking and visualization
  + Hyperparameter optimization with sweeps
  + Dataset analysis and exploration
* 🎨 **Clean Project Structure**
  + Modular and maintainable codebase
  + Easy to extend and customize

## 🛠️ Installation

```bash
# install dependencies
uv sync --dev
# install project as a package
uv pip install -e .
```

> [!note]
> 1. remember to replace `your_wandb_entity` with your actual W&B entity in the config files and `config.py`
> 2. `uv run` before bash scripts to ensure the environment is activated

## 📊 Dataset Analysis

Explore and analyze your dataset with built-in EDA tools:

```bash
python -m expt.main -c config/resnet.yml --eda
```

<details>
<summary>📈 View EDA Example</summary>

![EDA Example](assets/eda.png)

</details>

## 🚀 Training

Start training your model with a single command:

```bash
python -m expt.main -c config/resnet.yml --train
```

<details>
<summary>🔍 View Training Details</summary>

### Configuration Overview

![Config Display](assets/config_show.png)

### Training Progress

![Training Step](assets/step.png)

### Training Summary

![Training Summary](assets/summary.png)

### W&B Dashboard

![Wandb Dashboard](assets/train.png)

</details>

## 📊 Evaluation

Evaluate your trained model:

```bash
python -m expt.main -c config/resnet.yml --eval --run-id n8fjnlyi
```

<details>
<summary>📊 View Evaluation Results</summary>

![Evaluation Results](assets/result.png)

</details>

## 🎛️ Hyperparameter Tuning

Optimize your model with W&B Sweeps:

```bash
python -m expt.main -c config/resnet.yml --sweep --sweep-config config/sweep/mlp.yml
```

<details>
<summary>📈 View Sweep Results</summary>

![Sweep Results](assets/sweep.png)

</details>

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

* [PyTorch Lightning](https://lightning.ai/)
* [Weights & Biases](https://wandb.ai/)
* [Typer](https://typer.tiangolo.com/)
* [UV](https://github.com/astral-sh/uv)

---

<div align="center">
Made with ❤️ for the ML community
</div>
