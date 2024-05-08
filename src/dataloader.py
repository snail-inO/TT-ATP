#!/usr/bin/env python3

import pickle

from metamathpy.database import parse
from metamathpy.proof import verify_proof
import matplotlib.pyplot as pt

from utils import extract_proof_labels, build_tree, traverse_tree, max_value_len, TreeNode


db = parse("set.mm")

examples = []
goal_tokens, label_tokens = set(), set()
for r, rule in enumerate(db.rules.values()):
    if rule.consequent.tag != "$p":
        continue
    if len(rule.essentials) > 0:
        continue

    if len(examples) % 100 == 0:
        print(f"rule {rule.consequent.label} ({r} of {len(db.rules)})")

    goal = rule.consequent.tokens
    goal_tokens |= set(goal)
    goal = build_tree(goal)

    proof_root, _ = verify_proof(db, rule)
    if len(proof_root.dependencies) == 0:
        print(f"no proof for {rule.consequent.label}")
        continue
    labels = extract_proof_labels(proof_root)
    print(labels)
    label_tokens |= set(labels)
    examples.append((goal, labels))

print("raw examples:")
print("example 0:", examples[0])
print("example -10:", examples[-10])

# replace tokens with integer indexes
idx2goal = list(goal_tokens)
idx2label = list(label_tokens)

goal2idx = {token: t + 1 for t, token in enumerate(idx2goal)}
label2idx = {token: t for t, token in enumerate(idx2label)}

max_goal_len = max([max_value_len(goal) for goal, _ in examples])

examples = [
    (traverse_tree(goal, lambda x: goal2idx[x]), [label2idx[tok] for tok in labels])
    for (goal, labels) in examples
]

if __name__ == "__main__":
    print("indexed examples:")
    print("example 0:", examples[0])
    print("example -10:", examples[-10])
    print(f"{len(goal_tokens)} goal tokens, {len(label_tokens)} label tokens")
    print(f"max goal tree node size: {max_goal_len}")

    with open("dataset.pkl", "wb") as f:
        pickle.dump(examples, f)
        pickle.dump(idx2goal, f)
        pickle.dump(idx2label, f)
        pickle.dump(len(goal_tokens), f)
        pickle.dump(len(label_tokens), f)
        pickle.dump(max_goal_len, f)

    # pt.hist([len(proof) for (goal, proof) in examples], bins=100)
    # pt.xlabel("Proof length")
    # pt.ylabel("Frequency")
    # pt.yscale("log")
    # pt.savefig("proof_lengths.png")
