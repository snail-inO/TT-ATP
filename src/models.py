#!/usr/bin/env python3

import torch as tr

from utils import causal_mask, PositionalEncoding


class ChildSumTreeLSTMCell(tr.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ChildSumTreeLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_iou = tr.nn.Linear(input_size, 3 * hidden_size)
        self.U_iou = tr.nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.W_f = tr.nn.Linear(input_size, hidden_size)
        self.U_f = tr.nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, child_c, child_h):
        # Concatenate child hidden states
        child_h_sum = tr.sum(tr.stack(child_h), dim=0)

        # Compute input, output, and update gates
        iou = self.W_iou(x) + self.U_iou(child_h_sum)
        i, o, u = tr.chunk(iou, 3, dim=-1)
        i, o, u = tr.sigmoid(i), tr.sigmoid(o), tr.tanh(u)

        # Compute forget gate
        f = tr.sigmoid(self.W_f(x) + self.U_f(child_h_sum))

        # Compute cell state and hidden state
        c = tr.sum(f * tr.stack(child_c), dim=0) + i * u
        h = o * tr.tanh(c)

        return c, h


class ChildSumTreeLSTM(tr.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ChildSumTreeLSTM, self).__init__()
        self.cell = ChildSumTreeLSTMCell(input_size, hidden_size)
        self.hidden_size = hidden_size
        self.input_size = input_size

    def forward(self, node):
        cell_states, hidden_states = [], []
        if node.children == []:
            cell_state, hidden_state = self.cell(
                node.value.float(),
                [tr.zeros(self.input_size, self.hidden_size, device=node.value.device)],
                [tr.zeros(self.input_size, self.hidden_size, device=node.value.device)],
            )
        else:
            cells, hiddens = [], []
            for child in node.children:
                child_cell_states, child_hidden_states = self.forward(child)
                cells.extend(child_cell_states)
                hiddens.extend(child_hidden_states)

            # Compute cell state and hidden state using ChildSumTreeLSTMCell
            cell_state, hidden_state = self.cell(node.value.float(), cells, hiddens)

        # Append the computed states to the list
        cell_states.append(cell_state)
        hidden_states.append(hidden_state)

        return cell_states, hidden_states


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
            nhead=4,
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
        # goals = self.pos_enc(self.goal_embedding(goals[:, -self.max_len :]))

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
