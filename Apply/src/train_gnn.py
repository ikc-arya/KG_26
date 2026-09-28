"""Apply/GNN — LightGCN, hand-written on plain torch, evaluated with the shared evaluator.

LightGCN is the most stripped-down GNN that works for recommendation:
  E_0          = a free embedding per node (exactly like MF / KGE: a lookup table)
  E_{k+1}      = A_hat @ E_k            A_hat = D^-1/2 A D^-1/2  (symmetric-normalised)
  E_final      = mean(E_0, ..., E_K)    every node = blend of its 0..K-hop neighbourhood
  score(u, a)  = E_final[u] . E_final[a]
No weight matrices, no nonlinearity: the ONLY thing added over MF is neighbour averaging.
So `--layers 0` is plain matrix factorization -> the gap to K>0 is what message passing buys.

Graph: users <-> anime from train interactions (rated + watched), undirected.
  --content also adds anime <-> attribute edges (genre, studio, source, ...), so
  information can flow anime -> genre -> anime: the KG as extra paths. LightGCN ignores
  relation *types* (an edge is an edge); R-GCN / KGAT would keep them.
Loss: BPR, -log sigmoid(s(u, a_pos) - s(u, a_neg)), negative = random candidate anime.

Notebooks call `train(split, ...)` directly on a user subset (see gnn_trials.ipynb).

Run:  python src/train_gnn.py                 # LightGCN, 3 layers, interactions only
      python src/train_gnn.py --layers 0      # = matrix factorization (BPR)
      python src/train_gnn.py --content       # + KG content edges
"""

from __future__ import annotations

import argparse
import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from recsys_eval import GEN_MODELS, INTERACTIONS, Evaluator, load_split, save_result


def normalized_adjacency(src: torch.Tensor, dst: torch.Tensor, n: int) -> torch.Tensor:
    """Sparse D^-1/2 A D^-1/2 for an undirected graph given as one-way edge lists."""
    row = torch.cat([src, dst])
    col = torch.cat([dst, src])
    deg = torch.bincount(row, minlength=n).float()
    w = deg[row].rsqrt() * deg[col].rsqrt()  # 1/sqrt(d_i d_j): hubs don't drown the rest
    return torch.sparse_coo_tensor(torch.stack([row, col]), w, (n, n),
                                   check_invariants=True).coalesce()


class LightGCN(torch.nn.Module):
    def __init__(self, n_nodes: int, dim: int, layers: int, A: torch.Tensor):
        super().__init__()
        self.emb = torch.nn.Embedding(n_nodes, dim)
        torch.nn.init.normal_(self.emb.weight, std=0.1)
        self.layers, self.A = layers, A

    def propagate(self) -> torch.Tensor:
        e = self.emb.weight
        out = [e]
        for _ in range(self.layers):
            e = torch.sparse.mm(self.A, e)
            out.append(e)
        return torch.stack(out).mean(0)


@torch.no_grad()
def load_final_embeddings(name: str, split: dict[str, pd.DataFrame]) -> tuple[torch.Tensor, dict[str, int]]:
    """Checkpoint -> propagated E_final + node index. Rebuilds the graph the model was
    trained on (interactions, + content edges if it was a --content run)."""
    ck = torch.load(GEN_MODELS / f"{name}.pt")
    nodes = ck["nodes"]
    n2i = {x: i for i, x in enumerate(nodes)}
    train_df = split["train"]
    edges = train_df if ck["args"]["content"] else train_df[train_df.r.isin(INTERACTIONS)]
    A = normalized_adjacency(torch.tensor(edges.h.map(n2i).values),
                             torch.tensor(edges.t.map(n2i).values), len(nodes))
    model = LightGCN(len(nodes), ck["args"]["dim"], ck["args"]["layers"], A)
    model.emb.load_state_dict(ck["state"])
    return model.propagate(), n2i


def train(split: dict[str, pd.DataFrame], *, layers: int = 3, content: bool = False,
          dim: int = 64, epochs: int = 60, batch_size: int = 4096, lr: float = 5e-3,
          reg: float = 1e-4, eval_every: int = 5, patience: int = 3, seed: int = 26,
          verbose: bool = True) -> dict:
    """Train on split['train'], early-stop on valid, return test metrics of the best epoch."""
    torch.manual_seed(seed)
    ev = Evaluator(split)
    train_df = split["train"]
    inter = train_df[train_df.r.isin(INTERACTIONS)]
    edges = train_df if content else inter  # graph edges; training pairs are always `inter`

    nodes = sorted(set(edges.h) | set(edges.t))
    n2i = {x: i for i, x in enumerate(nodes)}
    A = normalized_adjacency(torch.tensor(edges.h.map(n2i).values),
                             torch.tensor(edges.t.map(n2i).values), len(nodes))
    U = torch.tensor(inter.h.map(n2i).values)
    I = torch.tensor(inter.t.map(n2i).values)
    cand = torch.tensor([n2i[a] for a in ev.candidates])

    model = LightGCN(len(nodes), dim, layers, A)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    E = None  # cached final embeddings for evaluation

    @torch.no_grad()
    def score_fn(users: list[str]) -> np.ndarray:
        return (E[[n2i[x] for x in users]] @ E[cand].T).numpy()

    log = print if verbose else (lambda *a, **k: None)
    log(f"{len(nodes):,} nodes, {A._nnz() // 2:,} undirected edges, "
        f"{len(U):,} training pairs, layers={layers}, content={content}")
    best, best_epoch, bad, history, best_state = -1.0, 0, 0, [], None
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(len(U))
        tot, nb = 0.0, 0
        for s in range(0, len(perm), batch_size):
            idx = perm[s:s + batch_size]
            u, i = U[idx], I[idx]
            j = cand[torch.randint(len(cand), (len(idx),))]

            E_all = model.propagate()  # full-graph propagation every step (graph is small)
            s_pos = (E_all[u] * E_all[i]).sum(-1)
            s_neg = (E_all[u] * E_all[j]).sum(-1)
            loss = -F.logsigmoid(s_pos - s_neg).mean()
            e0 = model.emb.weight
            l2 = reg * (e0[u].pow(2).sum(-1) + e0[i].pow(2).sum(-1)
                        + e0[j].pow(2).sum(-1)).mean()

            opt.zero_grad()
            (loss + l2).backward()
            opt.step()
            tot += loss.item()
            nb += 1

        # BPR loss is ln2 = 0.693 when pos and neg score the same (nothing learned)
        msg = f"epoch {epoch:>3}  loss {tot / nb:.4f}"
        if epoch % eval_every == 0:
            with torch.no_grad():
                E = model.propagate()
            v = ev.evaluate(score_fn, on="valid")[f"recall@{ev.k}"]
            history.append([epoch, round(tot / nb, 4), round(v, 4)])
            msg += f"  valid recall@{ev.k} {v:.4f}"
            if v > best:
                best, best_epoch, bad = v, epoch, 0
                best_state = copy.deepcopy(model.emb.state_dict())
            else:
                bad += 1
        log(msg, flush=True)
        if bad >= patience:
            log(f"early stop: no valid improvement for {patience} evals")
            break

    model.emb.load_state_dict(best_state)
    with torch.no_grad():
        E = model.propagate()
    return {"metrics": ev.evaluate(score_fn, on="test"), "best_epoch": best_epoch,
            f"best_valid_recall@{ev.k}": best, "history": history,
            "train_seconds": round(time.time() - t0),
            "model": model, "nodes": nodes, "E": E}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--content", action="store_true", help="add anime-attribute KG edges")
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--reg", type=float, default=1e-4, help="L2 on the batch's E_0 rows")
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--seed", type=int, default=26)
    p.add_argument("--name", default=None)
    args = p.parse_args()
    name = args.name or (f"lightgcn_k{args.layers}" + ("_content" if args.content else ""))
    kw = {k: v for k, v in vars(args).items() if k != "name"}

    out = train(load_split(), **kw)
    print(f"{name} (best epoch {out['best_epoch']}):", out["metrics"])
    GEN_MODELS.mkdir(parents=True, exist_ok=True)
    torch.save({"state": out["model"].emb.state_dict(), "nodes": out["nodes"],
                "args": vars(args)}, GEN_MODELS / f"{name}.pt")
    save_result(name, out["metrics"], {
        "args": vars(args), "best_epoch": out["best_epoch"],
        "best_valid_recall@10": out["best_valid_recall@10"], "history": out["history"],
        "train_seconds": out["train_seconds"],
    })


if __name__ == "__main__":
    main()
