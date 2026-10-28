# recsys

A self-directed technical deep-dive into recommender systems, in two parts.

| Project | What it covers |
|---|---|
| [`recsys_large_scale/`](recsys_large_scale/) | A full implicit-feedback pipeline on MovieLens 25M and the H&M Personalized Fashion dataset: popularity and item-item baselines, ALS and BPR matrix factorization, a FAISS-plus-LightGBM two-stage architecture, cold start and popularity bias, and a two-tower neural retrieval model. |
| [`recsys_benchmarks/`](recsys_benchmarks/) | External validation of those results: the same algorithms run through [RecBole](https://github.com/RUCAIBox/RecBole) on identical data and splits, a study of how much the evaluation protocol alone moves results, and a reproduction of the NCF paper's published setup. |

Each project has its own `README.md` with reproduction steps and its own `LEARNINGS.md`
with the write-up of what each experiment showed. `recsys_benchmarks/HIGHLIGHTS.md` is a
one-page summary of the benchmark findings.

## What is not in the repository

This repository holds source, configuration and write-ups only. Everything produced by
running the pipelines is excluded: raw and processed datasets, trained models, FAISS
indices, embedding dumps, result logs and generated charts. Together they run to roughly
14GB.

Each project's README documents how to obtain the data and regenerate the artifacts, and
the numbers themselves are reported in the `LEARNINGS.md` files.

## Getting started

Both projects use [uv](https://docs.astral.sh/uv/) and pin their own Python version:

```bash
cd recsys_large_scale && uv sync --group dev   # Python 3.14
cd recsys_benchmarks  && uv sync               # Python 3.11, required by RecBole
```

They are separate environments on purpose. RecBole 1.2.1 needs `numpy<2`, `pandas<3` and
`setuptools<81`, which the main project does not. `--group dev` installs pytest, which
the first step of `recsys_large_scale` uses to check the split logic.

Then follow `recsys_large_scale/README.md` from Phase 0. Each phase states the commands
to run and the output to check them against. `recsys_benchmarks/README.md` picks up from
there; a few of its steps run in the other project, and it says so where they do.
