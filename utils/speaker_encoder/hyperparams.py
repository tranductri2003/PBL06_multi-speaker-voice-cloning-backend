import os
from pathlib import Path

speakers_per_batch=32
utterances_per_speaker=10
seq_len=128
train_steps = 1e12
train_print_interval = 10 # in steps
total_evaluate_steps = 50
evaluate_interval = 500 # in steps
save_interval = 100 # in steps
save_dir = Path(os.environ.get('save_dir', None))
max_ckpts = 30
speaker_lr = 1e-4
libri_dataset_path = Path(os.environ.get('libri_dataset_path', None))
device = 'cuda:0'
loss_device = 'cpu'
