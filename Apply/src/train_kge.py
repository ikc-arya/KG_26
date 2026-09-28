"""Apply/embeddings — train a KGE model on the split, evaluate as a recommender.

Written by hand (not PyKEEN) so every moving part is visible. Two scoring functions,
everything else shared, so a difference in results is down to the score alone:
  DistMult  score(h, r, t) = sum_i h_i * r_i * t_i        relation SCALES each dim
  TransE    score(h, r, t) = gamma - ||h + r - t||_2      relation TRANSLATES
  loss           = softplus(-s_pos) + mean softplus(s_neg)  (logistic / BCE)
  negatives      = corrupt the tail with an entity from that relation's *range*
                   (anime for `rated`, genres for `hasGenre`, ...), so the model
                   learns to rank anime against anime, not anime against genres
  recommend      = score(user, rated, a) for every candidate anime a

Trains on ALL train triples (interactions + content edges) unless --no-content.

Notebooks call `train(split, ...)` directly on a user subset (see embeddings_trials.ipynb).

Run:  python src/train_kge.py                   # DistMult, defaults below
      python src/train_kge.py --model transe
      python src/train_kge.py --init-std 1e-3   # reproduces the Sep 23 ln 2 plateau
"""

from __future__ import annotations

import argparse
import copy
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from recsys_eval import GEN_MODELS, INTERACTIONS, Evaluator, load_split, save_result


class DistMult(torch.nn.Module):
    def __init__(self, n_ent: int, n_rel: int, dim: int, init_std: float):
        super().__init__()
        self.ent = torch.nn.Embedding(n_ent, dim)
        self.rel = torch.nn.Embedding(n_rel, dim)
        # Init scale matters more than usual: the score is a product of THREE
        # embeddings, so near zero both scores and gradients vanish (saddle).
        torch.nn.init.normal_(self.ent.weight, std=init_std)
        torch.nn.init.normal_(self.rel.weight, std=init_std)

    def score(self, h, r, t):
        return (self.ent(h) * self.rel(r) * self.ent(t)).sum(-1)

    def score_all(self, h, r, cand):
        """(len(h),) heads x one relation -> scores vs every candidate tail."""
        return (self.ent(h) * self.rel.weight[r]) @ self.ent(cand).T


class TransE(torch.nn.Module):
    def __init__(self, n_ent: int, n_rel: int, dim: int, init_std: float, gamma: float):
        super().__init__()
        self.ent = torch.nn.Embedding(n_ent, dim)
        self.rel = torch.nn.Embedding(n_rel, dim)
        torch.nn.init.normal_(self.ent.weight, std=init_std)
        torch.nn.init.normal_(self.rel.weight, std=init_std)
        # distance is >= 0, the logistic loss wants scores on both sides of 0:
        # gamma is the distance below which a triple counts as "true"
        self.gamma = gamma

    def score(self, h, r, t):
        return self.gamma - (self.ent(h) + self.rel(r) - self.ent(t)).norm(dim=-1)

    def score_all(self, h, r, cand):
        return self.gamma - torch.cdist(self.ent(h) + self.rel.weight[r], self.ent(cand))


def train(split: dict[str, pd.DataFrame], *, model: str = "distmult", dim: int = 128,
          epochs: int = 30, batch_size: int = 4096, lr: float = 0.01, negs: int = 16,
          init_std: float = 0.1, gamma: float = 6.0, reg: float = 1e-4,
          eval_every: int = 2, patience: int = 3, no_content: bool = False,
          neg_pool: str = "range", seed: int = 26, verbose: bool = True) -> dict:
    """Train on split['train'], early-stop on valid, return test metrics of the best epoch.

    neg_pool: "range" = corrupt tails from that relation's observed tails (default);
              "all"   = corrupt tails from any entity (the naive version).
    """
    torch.manual_seed(seed)
    ev = Evaluator(split)
    train_df = split["train"]
    if no_content:
        train_df = train_df[train_df.r.isin(INTERACTIONS)]

    # ---- index entities / relations ----
    ents = sorted(set(train_df.h) | set(train_df.t) | set(ev.candidates))
    e2i = {e: i for i, e in enumerate(ents)}
    rels = sorted(set(train_df.r))
    r2i = {r: i for i, r in enumerate(rels)}
    H = torch.tensor(train_df.h.map(e2i).values)
    R = torch.tensor(train_df.r.map(r2i).values)
    T = torch.tensor(train_df.t.map(e2i).values)

    # pool per relation: the tails a corrupted triple may draw from
    if neg_pool == "range":
        pools = [torch.tensor(sorted(set(T[R == ri].tolist()))) for ri in range(len(rels))]
    else:
        pools = [torch.arange(len(ents))] * len(rels)

    if model == "transe":
        net = TransE(len(ents), len(rels), dim, init_std, gamma)
    else:
        net = DistMult(len(ents), len(rels), dim, init_std)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    cand = torch.tensor([e2i[a] for a in ev.candidates])
    r_rated = r2i["rated"]

    @torch.no_grad()
    def score_fn(users: list[str]) -> np.ndarray:
        return net.score_all(torch.tensor([e2i[x] for x in users]), r_rated, cand).numpy()

    log = print if verbose else (lambda *a, **k: None)
    log(f"{len(ents):,} entities, {len(rels)} relations, {len(train_df):,} train triples")
    best, best_epoch, bad, history, best_state = -1.0, 0, 0, [], None
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(len(H))
        tot, nb = 0.0, 0
        for s in range(0, len(perm), batch_size):
            idx = perm[s:s + batch_size]
            h, r, t = H[idx], R[idx], T[idx]

            # tail corruption, sampled per relation from its pool
            t_neg = torch.empty(len(idx), negs, dtype=torch.long)
            for ri in r.unique().tolist():
                m = r == ri
                pool = pools[ri]
                t_neg[m] = pool[torch.randint(len(pool), (int(m.sum()), negs))]

            s_pos = net.score(h, r, t)
            s_neg = net.score(h[:, None], r[:, None], t_neg)
            loss = F.softplus(-s_pos).mean() + F.softplus(s_neg).mean()
            l2 = reg * (net.ent(h).pow(2).sum(-1) + net.rel(r).pow(2).sum(-1)
                        + net.ent(t).pow(2).sum(-1)).mean()

            opt.zero_grad()
            (loss + l2).backward()
            opt.step()
            tot += loss.item()
            nb += 1

        # the pos+neg loss sits at 2*ln2 = 1.386 when every score is 0 (DistMult: nothing learned)
        mean_loss = tot / nb
        msg = f"epoch {epoch:>3}  loss {mean_loss:.4f}"
        if epoch % eval_every == 0:
            v = ev.evaluate(score_fn, on="valid")[f"recall@{ev.k}"]
            history.append([epoch, round(mean_loss, 4), round(v, 4)])
            msg += f"  valid recall@{ev.k} {v:.4f}"
            if v > best:
                best, best_epoch, bad = v, epoch, 0
                best_state = copy.deepcopy(net.state_dict())
            else:
                bad += 1
        log(msg, flush=True)
        if bad >= patience:
            log(f"early stop: no valid improvement for {patience} evals")
            break

    # test with the best-on-valid weights
    net.load_state_dict(best_state)
    return {"metrics": ev.evaluate(score_fn, on="test"), "best_epoch": best_epoch,
            f"best_valid_recall@{ev.k}": best, "history": history,
            "train_seconds": round(time.time() - t0),
            "model": net, "ents": ents, "rels": rels}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["distmult", "transe"], default="distmult")
    p.add_argument("--gamma", type=float, default=6.0, help="TransE margin (score = gamma - dist)")
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--negs", type=int, default=16)
    p.add_argument("--init-std", type=float, default=0.1)
    p.add_argument("--reg", type=float, default=1e-4, help="L2 on the embeddings in each batch")
    p.add_argument("--eval-every", type=int, default=2)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--no-content", action="store_true", help="train on rated/watched only")
    p.add_argument("--neg-pool", choices=["range", "all"], default="range")
    p.add_argument("--seed", type=int, default=26)
    p.add_argument("--name", default=None, help="output name (default: --model)")
    args = p.parse_args()
    name = args.name or args.model
    kw = {k: v for k, v in vars(args).items() if k != "name"}

    out = train(load_split(), **kw)
    print(f"{name} (best epoch {out['best_epoch']}):", out["metrics"])
    GEN_MODELS.mkdir(parents=True, exist_ok=True)
    torch.save({"state": out["model"].state_dict(), "ents": out["ents"], "rels": out["rels"],
                "args": vars(args)}, GEN_MODELS / f"{name}.pt")
    save_result(name, out["metrics"], {
        "args": vars(args), "best_epoch": out["best_epoch"],
        "best_valid_recall@10": out["best_valid_recall@10"], "history": out["history"],
        "train_seconds": out["train_seconds"],
        "baseline_loss_at_zero_scores": round(2 * math.log(2), 4),
    })


if __name__ == "__main__":
    main()
