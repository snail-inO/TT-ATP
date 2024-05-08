#!/usr/bin/env python3

import pickle
import random

import numpy as np
import torch as tr

from models import ChildSumTreeLSTM, Prooformer
from utils import tersorize_tree

d_model = 256
max_len = 2048
num_layers = 4

dev = tr.device("cuda") if tr.cuda.is_available() else tr.device("cpu")

with open("dataset.pkl", "rb") as f:
    examples = pickle.load(f)
    idx2goal = pickle.load(f)
    idx2label = pickle.load(f)
    len_goal_tokens = pickle.load(f)
    len_label_tokens = pickle.load(f)
    max_goal_len = pickle.load(f)

print(f"dataset size = {len(examples)}")
data_size = 1000
examples = examples[:data_size]
random.seed(8762)
random.shuffle(examples)

train_size = int(0.7 * len(examples))
val_size = int(0.15 * len(examples))

train_samples = examples[:train_size]
test_samples = examples[train_size:train_size + val_size]
val_samples = examples[train_size + val_size:]

embedder = ChildSumTreeLSTM(max_goal_len, d_model).to(dev)
model = Prooformer(
    d_model, max_len, num_layers, len_goal_tokens, len_label_tokens
).to(dev)

embedder.load_state_dict(tr.load("embedder0.pth"))
model.load_state_dict(tr.load("model0.pth"))

embedder.eval()
model.eval()

loss_fn = tr.nn.CrossEntropyLoss()

test_loss, test_accu = [], []
for b, (goal, proof) in enumerate(test_samples):

    # forward
    goal_copied = goal.copy()
    proofs = tr.tensor([proof]).to(dev)
    # goals = tr.tensor([goal]).to(dev)
    goals = [tersorize_tree(goal_copied, max_goal_len, dev)]
    embedded_goals = []
    
    for goal in goals:
        _, hidden_states = embedder(goal)
        embedded_goals.append(hidden_states[-1])
    embedded_goals = tr.stack(embedded_goals)
    
    proofs, targs = proofs[:, :-1], proofs[:, 1:]
    logits = model(proofs, embedded_goals)
    # logits = model(proofs, goals)

    # loss
    logits = logits.flatten(end_dim=-2)  # flatten batch and sequence dims
    targs = targs[:, -max_len:].flatten()  # flatten batch and sequence dims
    test_loss.append(loss_fn(logits, targs).item())
    test_accu.append((logits.argmax(dim=-1) == targs).to(float).mean().item())

print(f"\n***test loss = {np.mean(test_loss)}, accu = {np.mean(test_accu)}\n")

with open(f"test_result_0.pkl", "wb") as f:
    pickle.dump(test_loss, f)
    pickle.dump(test_accu, f)