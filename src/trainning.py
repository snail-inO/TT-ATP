#!/usr/bin/env python3

import random
import pickle

import torch as tr
import numpy as np
import matplotlib.pyplot as pt

from models import Prooformer, ChildSumTreeLSTM
from utils import tersorize_tree, traverse_tree

from time import perf_counter

dev = tr.device("cuda") if tr.cuda.is_available() else tr.device("cpu")
print(f"device = {dev}")

with open("dataset.pkl", "rb") as f:
    examples = pickle.load(f)
    idx2goal = pickle.load(f)
    idx2label = pickle.load(f)
    len_goal_tokens = pickle.load(f)
    len_label_tokens = pickle.load(f)
    max_goal_len = pickle.load(f)

data_size = 1000
examples = examples[:data_size]
# random.shuffle(examples)
train_size = int(0.9 * len(examples))
train_samples, test_samples = examples[:-train_size], examples[-train_size:]

# for goal, labels in test_samples:
#   print([idx2goal[tok] for tok in goal], [idx2label[tok] for tok in labels])
# input('.')

num_updates = 10000
test_period = 500
d_model = 256
max_len = 2048
num_layers = 2
batch_size = 1

embedder = ChildSumTreeLSTM(max_goal_len, d_model).to(dev)
model = Prooformer(
    d_model, max_len, num_layers, len_goal_tokens, len_label_tokens
).to(dev)
loss_fn = tr.nn.CrossEntropyLoss()
opt = tr.optim.Adam(model.parameters(), lr=0.00005)

start = perf_counter()
loss_curve = []
accu_curve = []
example_idx = 0
for update in range(num_updates):

    # prepare training batch
    proofs, goals = [], []
    for b in range(batch_size):
        example = train_samples[example_idx % len(train_samples)]
        example_idx += 1
        if not isinstance(example[0].value, tr.Tensor):
            tersorize_tree(example[0], max_goal_len, dev)
        goals.append(example[0])
        proofs.append(example[1])

    # forward
    proofs = tr.tensor(proofs).to(dev)
    full_proofs = proofs
    proofs, targs = proofs[:, :-1], proofs[:, 1:]
    embedded_goals = []
    for goal in goals:
        _, hidden_states = embedder(goal)
        embedded_goals.append(hidden_states[-1])
    embedded_goals = tr.stack(embedded_goals)
    logits = model(proofs, embedded_goals)

    # loss
    logits = logits.flatten(end_dim=-2)  # flatten batch and sequence dims
    targs = targs[:, -max_len:].flatten()  # flatten batch and sequence dims
    loss = loss_fn(logits, targs)

    # print(goals, proofs, targs)

    # backward
    opt.zero_grad()
    loss.backward()
    opt.step()

    # progress
    loss_curve.append(loss.item())
    accu_curve.append((logits.argmax(dim=-1) == targs).to(float).mean().item())
    if update % 100 == 0:
        print(f"update {update}: loss={loss_curve[-1]}, accu={accu_curve[-1]}")

    # test
    if update % test_period != 0:
        continue

    embedder.eval()
    model.eval()

    test_loss, test_accu = [], []
    for b, (goal, proof) in enumerate(test_samples):

        # forward
        goal_copied = goal.copy()
        proofs = tr.tensor([proof]).to(dev)
        goals = [tersorize_tree(goal_copied, max_goal_len, dev)]
        embedded_goals = []
        
        for goal in goals:
            _, hidden_states = embedder(goal)
            embedded_goals.append(hidden_states[-1])
        embedded_goals = tr.stack(embedded_goals)
        
        proofs, targs = proofs[:, :-1], proofs[:, 1:]
        logits = model(proofs, embedded_goals)

        # loss
        logits = logits.flatten(end_dim=-2)  # flatten batch and sequence dims
        targs = targs[:, -max_len:].flatten()  # flatten batch and sequence dims
        test_loss.append(loss_fn(logits, targs).item())
        test_accu.append((logits.argmax(dim=-1) == targs).to(float).mean().item())

        if np.isnan(test_loss[-1]):
            temp_goal = goal.copy()
            traverse_tree(temp_goal, lambda x: idx2goal[x])
            print(temp_goal)
            # print(test_loss)
            # print(test_accu)
            print(proofs)
            print(targs)
            print()
            # input(".")

    print(f"\n***test loss = {np.mean(test_loss)}, accu = {np.mean(test_accu)}\n")

    embedder.train()
    model.train()


tr.save(embedder.state_dict(), "embedder2.pth")
tr.save(model.state_dict(), "model2.pth")
print(f"total time = {perf_counter()-start}s")

pt.subplot(1,2,1)
pt.plot(loss_curve[::10])
pt.ylabel("Loss")
pt.xlabel("Update")
pt.subplot(1,2,2)
pt.plot(accu_curve[::20])
pt.ylabel("Accuracy")
pt.xlabel("Update")
pt.savefig("training_curve4.png")