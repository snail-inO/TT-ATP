#!/usr/bin/env python3


from metamathpy.database import parse
from metamathpy.proof import verify_proof
import matplotlib.pyplot as pt

from utils import extract_proof_labels


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

    proof_root, _ = verify_proof(db, rule)
    labels = extract_proof_labels(proof_root)
    label_tokens |= set(labels)

    examples.append((goal, labels))
    if len(examples) == 1000:
        break

print("raw examples:")
print("example 0:", examples[0])
print("example -1:", examples[-1])

# replace tokens with integer indexes
idx2goal = list(goal_tokens)
idx2label = list(label_tokens)

goal2idx = {token: t for t, token in enumerate(idx2goal)}
label2idx = {token: t for t, token in enumerate(idx2label)}

examples = [
    ([goal2idx[tok] for tok in goal], [label2idx[tok] for tok in labels])
    for (goal, labels) in examples
]

if __name__ == "__main__":
    print("indexed examples:")
    print("example 0:", examples[0])
    print("example -1:", examples[-1])
    print(f"{len(goal_tokens)} goal tokens, {len(label_tokens)} label tokens")
    print(goal_tokens)
    print(label_tokens)

    pt.hist([len(proof) for (goal, proof) in examples], bins=100)
    pt.xlabel("Proof length")
    pt.ylabel("Frequency")
    pt.yscale("log")
    pt.savefig("proof_lengths.png")
