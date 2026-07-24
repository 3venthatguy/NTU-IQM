# Known Discrepancies

> **Audience:** both. **Purpose:** where existing artifacts disagree with the current code — so nobody is misled. Each item is "if you see X, know that Y."
> ⬅ Back to [docs hub](README.md)

These are **not bugs to fix here** — they're drift between older docs/scripts and the current nanotube-only code. The [component docs](README.md) describe current reality; this page catalogs the traps.

## 1. Top-level `README.md` describes the *old* 2D-lattice SCIGEN
**If you see** the top-level [`README.md`](../README.md) listing constraints like Triangular / Honeycomb / Kagome / Square / Lieb, and a `sc_dict = {'tri': SC_Triangular, 'hon': SC_Honeycomb, …}` (around line 234),
**know that** those classes **no longer exist**. The current code is nanotube-only:
```python
sc_dict = {'ntb': SC_Nanotube, 'cnt': SC_CarbonTube, 'alx': SC_DBTemplate, 'van': SC_Vanilla}
```
The README documents upstream SCIGEN, not NTGEN. Trust [components/structural-constraints.md](components/structural-constraints.md) and [`script/sc_utils.py`](../script/sc_utils.py) instead. (The README is still useful for dependency versions and the general framework description.)

## 2. `screen_mul.py` `sc_list` uses stale 2D names
**If you see** `screen_mul.py` with
```python
sc_list = ['tri', 'hon', 'kag', 'sqr', 'elt', 'sns', 'tsq', 'srt', 'snh', 'trh', 'lieb']
```
**know that** these are the old 2D-lattice motif names and don't match the nanotube constraints. This batch-screening driver was inherited and not updated. For screening, prefer calling `script/eval_screen.py --label <label>` directly on your generation output (which is constraint-agnostic — it screens whatever structures are in the `.pt`). → [components/gnn-screening.md](components/gnn-screening.md).

## 3. Sibling notebooks assume a `NTGEN-edit` path
**If you see** a notebook in `../comp_models/NTGEN_generation/` (e.g. `05_ctgen_generation.ipynb`, line ~315: `cd models/NTGEN-edit`) referencing `../NTGEN-edit`, `models/NTGEN-edit`, or cloning into such a folder,
**know that** this codebase was renamed to `NTGENS/` and moved to the repo root. Those relative paths are stale. When running such a notebook, point it at the current `NTGENS/` location. Also note `RETRAIN_CARBON.md` refers to `../CTGEN_generation/…`, another old name for the notebook folder now called `NTGEN_generation`.

> **Fixed in `ntgen_generation.ipynb`** (updated alongside this doc): its `PROJECT_DIR`/`NOTEBOOK_DIR` now point at `NTGENS/` and `comp_models/NTGEN_generation/`. `05_ctgen_generation.ipynb` still has the one stale `cd models/NTGEN-edit` reference above — fix it the same way if you touch that notebook.

## 4. `NTGENS/` is untracked (git sees the old layout)
**If you see** `git status` showing `NTGENS/` as untracked (`??`) and many `models/NTGEN-edit/...` paths as deleted (`D`),
**know that** the working tree was reorganized (`models/NTGEN-edit/ → NTGENS/`, `models/DiffCSP-main/ + models/NTGEN_generation/ → comp_models/`) but **not yet committed**. `NTGENS/` is byte-identical to `models/NTGEN-edit@HEAD`. Any file/line reference in these docs was checked against the working tree — if git history looks different, that's why. Committing the rename would resolve this.

## 5. No `requirements.txt` and no test suite
**If you see** `pytest` in the dependency list, or expect a `tests/` directory,
**know that** there is **no test suite, no `conftest.py`, no `test_*.py`** anywhere, and **no `requirements.txt`/`environment.yml`** — dependencies live only as prose in the two READMEs (consolidated in [usage/setup.md](usage/setup.md)). The closest things to tests are `--cfg job` smoke checks, `if __name__ == '__main__'` Hydra stubs, and the manual "set `sc_list` and run `gen_mul.py`" workflow. See the stub-testing recipe in [usage/extending.md](usage/extending.md).

## 6. Stray LLM-prompt comment in `gen_utils.py`
**If you see** a comment in [`script/gen_utils.py`](../script/gen_utils.py) (just above `parse_none_or_value`, ~line 105) that reads like a pasted chat prompt ("I am trying to pass None keyword as a command line parameter … Please write a function that takes a list of strings …"),
**know that** it is an **accidentally-committed artifact**, inert and unrelated to the code below it. The `parse_none_or_value(argument, obj=float)` function itself *is* real and used (it lets CLI args pass the string `'None'` through as Python `None`). Safe to ignore the comment; safe to delete it if cleaning up.

## 7. Config defaults ≠ used values
**If you see** `conf/default.yaml` composing `model: diffusion` and `data: default`,
**know that** the **used** model is `diffusion_w_type` and there is **no** `conf/data/default.yaml` (only `carbon_24.yaml` / `mp_20.yaml`). Every real command overrides both: `python scigen/run.py data=mp_20 model=diffusion_w_type expname=…`. Running with the bare defaults will fail on the missing data config. → [components/configuration.md](components/configuration.md).

## Summary table

| # | Artifact | Says | Reality |
|---|---|---|---|
| 1 | top-level `README.md` | 2D lattice constraints | nanotube-only `sc_dict` |
| 2 | `screen_mul.py` | `tri/hon/kag/…` | those don't exist |
| 3 | `comp_models/NTGEN_generation/*.ipynb` | `NTGEN-edit` paths | folder is now `NTGENS/` |
| 4 | `git status` | old `models/…` layout | `NTGENS/` untracked, = `NTGEN-edit@HEAD` |
| 5 | dep list mentions `pytest` | tests exist | no tests, no requirements file |
| 6 | comment in `gen_utils.py` | (chat prompt) | accidental noise, ignore |
| 7 | `conf/default.yaml` | `model: diffusion`, `data: default` | use `diffusion_w_type` + a real dataset |

## Next

- Back to the [docs hub](README.md)
- Avoid these while extending → [usage/extending.md](usage/extending.md)
