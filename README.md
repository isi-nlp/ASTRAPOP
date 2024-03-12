# ASTRAPOP
The official repository for the paper "Authorship Style Transfer with Policy Optimization".

# Installation

Commends for enviroment setup with conda.
```bash
conda create --name astrapop python=3.8
conda activate astrapop
pip install -U pip
pip install -r requirements.txt
```

# Data

Please download the original Reddit Million User Dataset (MUD) from [here](https://github.com/noa/naacl2021) and the original ETS Corpus of Non-Native Written English from [here](https://catalog.ldc.upenn.edu/LDC2014T06). We will publish the data preprocessing code soon.

# Reproduce Results
## Reddit
To reproduce the results on the Reddit dataset, please run the scirpts in `scripts/reddit` following the procedure below.
1. Train the paraphrase model and the reference SFT model by running `00_train_paraphraser.sh` and `00_train_sft.sh`.
2. Generate the data for DPO and CPO training by running `01_generate_dpo_cpo_data.sh`.
3. Train the PO models using PPO/DPO/CPO by running `02_train_ppo.sh`/`02_train_dpo.sh`/`02_train_cpo.sh`.
4. Transfer the texts in the test set by running `03_generate.sh`.
## ETS
To reproduce the results on the ETS dataset, please run the scirpts in `scripts/ets`.
1. Train the style reward model, the paraphrase model, and the reference SFT model by running `00_train_cls.sh`, `00_train_paraphraser.sh`, and `00_train_sft.sh`.
2. Generate the data for DPO and CPO training by running `01_generate_dpo_cpo_data.sh`.
3. Train the PO models using PPO/DPO/CPO by running `02_train_ppo.sh`/`02_train_dpo.sh`/`02_train_cpo.sh`.
4. Transfer the texts in the test set by running `03_generate.sh`.
