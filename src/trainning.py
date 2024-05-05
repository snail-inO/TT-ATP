#!/usr/bin/env python3

import torch as tr
import numpy as np
import matplotlib.pyplot as pt
import random

from models import Prooformer
from dataloader import examples, goal_tokens, label_tokens, idx2goal, idx2label
from time import perf_counter

dev = tr.device("cuda") if tr.cuda.is_available() else tr.device("cpu")
print(f"device = {dev}")

random.shuffle(examples)
train_samples, test_samples = examples[:-10], examples[-10:]

# for goal, labels in test_samples:
#   print([idx2goal[tok] for tok in goal], [idx2label[tok] for tok in labels])
# input('.')

num_updates = 10000
test_period = 500
d_model = 256
max_len = 2048
num_layers = 2
batch_size = 1

model = Prooformer(
    d_model, max_len, num_layers, len(goal_tokens), len(label_tokens)
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

        goals.append(example[0])
        proofs.append(example[1])

    # forward
    goals, proofs = tr.tensor(goals).to(dev), tr.tensor(proofs).to(dev)
    full_proofs = proofs
    proofs, targs = proofs[:, :-1], proofs[:, 1:]
    logits = model(proofs, goals)

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

    model.eval()

    test_loss, test_accu = [], []
    for b, (goal, proof) in enumerate(test_samples):

        # forward
        goals, proofs = tr.tensor([goal]).to(dev), tr.tensor([proof]).to(dev)
        proofs, targs = proofs[:, :-1], proofs[:, 1:]
        logits = model(proofs, goals)

        # loss
        logits = logits.flatten(end_dim=-2)  # flatten batch and sequence dims
        targs = targs[:, -max_len:].flatten()  # flatten batch and sequence dims
        test_loss.append(loss_fn(logits, targs).item())
        test_accu.append((logits.argmax(dim=-1) == targs).to(float).mean().item())

        if np.isnan(test_loss[-1]):
            print([idx2goal[tok] for tok in goal])
            print(test_loss, test_accu)
            print(goals, proofs, targs)
            input(".")

    print(f"\n***test loss = {np.mean(test_loss)}, accu = {np.mean(test_accu)}\n")

    model.train()
tr.save(model.state_dict(), "model.pth")
print(f"total time = {perf_counter()-start}s")

pt.subplot(1,2,1)
pt.plot(loss_curve[::10])
pt.ylabel("Loss")
pt.xlabel("Update")
pt.subplot(1,2,2)
pt.plot(accu_curve[::20])
pt.ylabel("Accuracy")
pt.xlabel("Update")
pt.savefig("training_curve.png")