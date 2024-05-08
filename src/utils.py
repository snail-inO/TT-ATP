#!/usr/bin/env python3

import torch as tr
import numpy as np
import matplotlib.pyplot as pt

from metamathpy.database import parse
from metamathpy.proof import verify_proof


def causal_mask(sz):
    return tr.log(tr.tril(tr.ones(sz, sz)))


def pe_tensor(d_model, max_len, base):
    pe = tr.zeros(max_len, d_model)
    position = tr.arange(0, max_len, dtype=tr.float).unsqueeze(1)
    div_term = tr.exp(tr.arange(0, d_model, 2).float() * (-np.log(base) / d_model))
    pe[:, 0::2] = tr.sin(position * div_term)
    pe[:, 1::2] = tr.cos(position * div_term)
    return pe


class PositionalEncoding(tr.nn.Module):

    def __init__(self, d_model, max_len, base):
        super(PositionalEncoding, self).__init__()
        self.max_len = max_len
        self.register_buffer("pe", pe_tensor(d_model, max_len, base))

    def forward(self, x):
        seq_len = min(self.max_len, x.shape[1])
        x = x[:, -seq_len:] + self.pe[:seq_len, :]
        return x


def extract_proof_labels(proof_root):
    labels = []
    rule = proof_root.rule
    if len(proof_root.dependencies) > 0:
        # for hyp in rule.essentials:
        for hyp in rule.floatings + rule.essentials:
            dep = proof_root.dependencies[hyp.label]
            sublabels = extract_proof_labels(dep)
            labels.extend(sublabels)

    labels.append(rule.consequent.label)
    return labels

class TreeNode:
    def __init__(self):
        self.value = []
        self.children = []
    def __repr__(self):
        return f"TreeNode({self.value}, {self.children})"
    def __eq__(self, other):
        return self.value == other.value and self.children == other.children
    def copy(self):
        new_node = TreeNode()
        new_node.value = self.value.copy()
        new_node.children = [child.copy() for child in self.children]
        return new_node
    
def build_tree(expression):
    def helper(it):
        node = TreeNode()
        for token in it:
            if token == '(':
                node.children.append(helper(it))
            elif token == ')':
                return node
            else:
                node.value.append(token)
        return node

    return helper(iter(expression))

def traverse_tree(node, f):
    node.children = [traverse_tree(child, f) for child in node.children]
    for i, token in enumerate(node.value):
        node.value[i] = f(token)
    return node

def tersorize_tree(node, size, dev):
    node.children = [tersorize_tree(child, size, dev) for child in node.children]
    node.value = tr.tensor(node.value).clone().detach().to(dev)

    # Calculate the required padding
    padding = size - node.value.size(0)
    
    # Pad the tensor
    node.value = tr.nn.functional.pad(node.value, (0, max(padding, 0)))
    return node
    
def max_value_len(node):
    return max([max_value_len(child) for child in node.children] + [len(node.value)])

if __name__ == "__main__":
    # Test causal_mask
    mask = causal_mask(10)
    pt.imshow(mask)
    pt.savefig("causal_mask.png")

    # Test pe_tensor
    pe = pe_tensor(d_model=256, max_len=128, base=10000)
    pt.imshow(pe @ pe.t())
    pt.savefig("pe.png")

    x = tr.randn(2, 3, 6)
    x = PositionalEncoding(6, 3, 10)(x)
    print(x)

    # Test extract_proof_labels
    db = parse("set.mm")
    root, _ = verify_proof(db, db.rules["mpd"])
    print(root)
    print(root.dependencies)
    print(extract_proof_labels(root))
