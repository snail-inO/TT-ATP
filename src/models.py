#!/usr/bin/env python3

import torch as tr

from utils import causal_mask, PositionalEncoding


class Prooformer(tr.nn.Module):
    def __init__(self, d_model, max_len, num_layers, num_goal_tokens, num_proof_tokens):
        super().__init__()
        self.max_len = max_len
        self.goal_embedding = tr.nn.Embedding(
            num_embeddings=num_goal_tokens, embedding_dim=d_model
        )
        self.proof_embedding = tr.nn.Embedding(
            num_embeddings=num_proof_tokens, embedding_dim=d_model
        )
        self.pos_enc = PositionalEncoding(d_model, max_len, 10000)
        self.trf = tr.nn.Transformer(
            d_model,
            nhead=8,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            batch_first=True,
        )

    def forward(self, proofs, goals):
        """
        proofs[b,t]: tth token of bth proof in the batch
        goals[b,s]: sth token of bth goal in the batch
        logits[b,t,r]: logit for label r at tth step of bth example of the batch
        """

        # embed and position encode
        proofs = self.pos_enc(self.proof_embedding(proofs[:, -self.max_len :]))
        goals = self.pos_enc(self.goal_embedding(goals[:, -self.max_len :]))

        # transformer
        mask = causal_mask(proofs.shape[1])
        result = self.trf(goals, proofs, tgt_mask=mask)

        # get logits for next proof label
        readout = self.proof_embedding.weight
        logits = result @ readout.t()
        return logits


if __name__ == "__main__":
    # Test Prooformer
    model = Prooformer(
        d_model=64, max_len=100, num_layers=2, num_goal_tokens=10, num_proof_tokens=20
    )

    proofs = tr.randint(20, (2, 200))
    goals = tr.randint(10, (2, 50))
    logits = model(proofs, goals)
    print(logits)
    print(logits.shape)
