# AQME pipeline: what it does, why, and how to use it

All AQME-related files live in **deepchem/aqme/**. The pipeline reads **data_mingap.csv** from the parent folder (**deepchem/**) and writes **aqme_input.csv**, **CSEARCH/**, **QCALC_***, and **SP_*** inside this folder. The original dataset is never modified.

This pipeline turns your **data_mingap.csv** (Diels–Alder reactions with diene/dienophile IDs and pre-computed ΔG) into **AQME-ready input** and then **Gaussian job files** so you can recompute energies at a chosen level of theory in **gas** or **SMD/toluene**.

---

## Folder layout

```
deepchem/
├── data_mingap.csv          ← read by converter (not modified)
├── aqme/
│   ├── README.md            ← this file
│   ├── aqme_converter.py    ← converts data_mingap.csv → aqme_input.csv
│   ├── run_aqme_pipeline.py ← CSEARCH, QPREP, QCORR
│   ├── aqme_input.csv       ← written here (AQME input)
│   ├── CSEARCH/             ← conformers (SDFs)
│   ├── QCALC_gas/           ← Gaussian .com (gas)
│   ├── QCALC_toluene/       ← Gaussian .com (SMD/toluene)
│   ├── SP_gas/              ← single-point .com (after QCORR)
│   └── SP_toluene/
└── ...
```

---

## What the pipeline does

1. **Convert** `data_mingap.csv` → **aqme_input.csv**  
   Turns your reaction table (diene/dienophile IDs) into the format AQME expects: SMILES, species names, and (for TSs) distance constraints for the two forming C–C bonds. Reads from **deepchem/data_mingap.csv**, writes **deepchem/aqme/aqme_input.csv**.

2. **Conformational search (CSEARCH)**  
   Uses AQME + CREST to generate conformers (and TS guesses) from the SMILES in `aqme_input.csv`. Output is 3D structures in **aqme/CSEARCH/** (SDFs).

3. **Write Gaussian inputs (QPREP)**  
   From those SDFs, writes Gaussian **.com** files in **aqme/QCALC_gas/** or **aqme/QCALC_toluene/** for:
   - **Opt + freq** at M06-2X/def2-TZVP (gas or SMD/toluene), with TS-specific keywords for transition states.
   - Optionally, after you run Gaussian and QCORR, **single-point** at M06-2X/6-31G(d) in **aqme/SP_gas/** or **aqme/SP_toluene/** (for SPC/GoodVibes).

4. **Check/fix outputs (QCORR)**  
   After you run Gaussian, QCORR checks the LOG files (convergence, imaginary frequencies, etc.) and can suggest or write fixed inputs.

You run **Gaussian** (and later **GoodVibes**) yourself; the pipeline only prepares inputs and runs AQME’s CSEARCH, QPREP, and QCORR steps.

---

## Logic behind the flow

| Step | Why it’s there |
|------|-----------------|
| **data_mingap.csv** | Your source of truth: one row per reaction (diene + dienophile IDs). No SMILES, no 3D. Lives in **deepchem/**. |
| **aqme_input.csv** | AQME needs SMILES and names (and TS constraints). The converter uses the same ID→SMILES mapping as `reaction_optimizer.id_to_smiles()` so species match your ML setup. TS rows get a `constraints_dist` for the two forming bonds (C2–C5, C3–C6 at 2.35 Å) so CREST can search for the TS. Written in **aqme/**. |
| **CSEARCH** | Conformers/TS guesses from SMILES. One CSEARCH run is shared for both gas and toluene; only the later Gaussian route lines change. Output in **aqme/CSEARCH/**. |
| **QPREP opt+freq** | Produces the .com files you actually run in Gaussian. Solvent is chosen here: **gas** = no `scrf=`, **toluene** = `scrf=(smd,solvent=toluene)`. Output in **aqme/QCALC_***. |
| **You run Gaussian** | The pipeline does not execute Gaussian; you run the .com jobs on your machine or cluster. |
| **QCORR** | Validates and corrects LOGs (e.g. extra imaginary frequencies, convergence issues) so only good structures go forward. |
| **QPREP SP** | Builds single-point .com at 6-31G(d) for GoodVibes-style thermochemistry (same solvent as opt+freq). Output in **aqme/SP_***. |

So: **IDs → SMILES → 3D (CSEARCH) → Gaussian inputs (QPREP) → you run Gaussian → QCORR → optional SP inputs → you run Gaussian again → GoodVibes.**

---

## Level of theory

- **Geometry + frequencies:** M06-2X/def2-TZVP  
  - Gas: no solvent.  
  - Toluene: `scrf=(smd,solvent=toluene)`.
- **Single-point (for SPC):** M06-2X/6-31G(d), same environment (gas or toluene).

So the composite is **M06-2X/def2-TZVP // M06-2X/6-31G(d)**; with toluene it’s **SMD** for both steps.

---

## How to use it

### Option A: Click Run (no terminal)

Run the scripts from **deepchem/aqme/** (open the folder in your IDE or run from there). All behaviour is controlled by **defaults defined in the scripts**.

**1. Converter (`aqme_converter.py`)**

- **Click Run** on `aqme_converter.py` (from the **aqme** folder).  
- It reads **../data_mingap.csv** and writes **aqme_input.csv** in this folder.  
- To change input/output or whether reactant rows are included, edit at the top:
  - `DEFAULT_DATA_FILE` (name in deepchem/),
  - `DEFAULT_OUTPUT_FILE` (name in aqme/),
  - `INCLUDE_REACTANTS`

**2. Pipeline (`run_aqme_pipeline.py`)**

- **Click Run** on `run_aqme_pipeline.py` (with no command-line arguments).  
- It uses the in-file defaults. By default it will:
  - Ensure **aqme_input.csv** exists (run the converter logic if needed),
  - Run **CSEARCH**,
  - Run **QPREP** for opt+freq for the default solvent.

Edit the block at the top of `run_aqme_pipeline.py` to control what runs and for which solvent:

| Variable | Meaning | Typical use |
|----------|--------|-------------|
| `DEFAULT_SOLVENT` | `"gas"` or `"toluene"` | Choose gas or SMD/toluene for all generated .com files. |
| `DEFAULT_RUN_CONVERTER` | Create/refresh `aqme_input.csv` from `data_mingap.csv` | `True` the first time or when you change the source data. |
| `DEFAULT_RUN_CSEARCH` | Run conformational search | `True` to (re)generate `CSEARCH/` SDFs. |
| `DEFAULT_RUN_QPREP_OPT` | Write Gaussian opt+freq .com files | `True` to (re)generate `QCALC_gas/` or `QCALC_toluene/`. |
| `DEFAULT_RUN_QCORR` | Run QCORR on existing LOGs | Set to `True` **after** you have run Gaussian on the QCALC .com files. |
| `DEFAULT_RUN_QPREP_SP` | Write single-point .com files | Set to `True` **after** QCORR, when you want SP jobs for GoodVibes. |
| `DEFAULT_NPROCS`, `DEFAULT_MEM`, `DEFAULT_WORKDIR` | Resources and working directory | `DEFAULT_WORKDIR` is this folder (**aqme/**); adjust others if needed. |

**Typical sequence when using “Run” only:**

1. Run **aqme_converter.py** once (or leave `DEFAULT_RUN_CONVERTER = True` and run the pipeline once).
2. Run **run_aqme_pipeline.py** → creates **aqme/CSEARCH/** and **aqme/QCALC_<solvent>/** with .com files.
3. Run **Gaussian** on all .com files in **aqme/QCALC_<solvent>/**.
4. Set `DEFAULT_RUN_QCORR = True`, run **run_aqme_pipeline.py** again.
5. Set `DEFAULT_RUN_QPREP_SP = True`, run **run_aqme_pipeline.py** again → creates **aqme/SP_<solvent>/** .com files.
6. Run **Gaussian** on the SP .com files, then use **GoodVibes** (or your workflow) with the opt+freq and SP LOGs.

You can set `DEFAULT_RUN_QCORR` and `DEFAULT_RUN_QPREP_SP` to `False` when you only want to regenerate .com files (e.g. after changing `DEFAULT_SOLVENT`).

### Option B: Terminal with arguments

From **deepchem/aqme/**:

```bash
cd deepchem/aqme

# 1. Create aqme_input.csv
python aqme_converter.py

# 2. Full prep for gas (CSEARCH + QPREP opt+freq)
python run_aqme_pipeline.py --solvent gas

# 2b. Same for toluene
python run_aqme_pipeline.py --solvent toluene

# 3. After running Gaussian on QCALC_*:
python run_aqme_pipeline.py --solvent gas --qcorr

# 4. After QCORR, prepare single-point inputs
python run_aqme_pipeline.py --solvent gas --qprep-sp
```

You can also run only specific steps (e.g. `--csearch`, `--qprep-opt`) and combine with `--nprocs`, `--mem`, `--workdir` as needed.

---

## Output layout (all under deepchem/aqme/)

| File or folder | Description |
|----------------|-------------|
| **aqme_input.csv** | New dataset: SMILES, code_name, constraints_dist (from data_mingap.csv). Not modified by later steps. |
| **CSEARCH/** | Conformers/TS guesses (SDFs). Shared by gas and toluene. |
| **QCALC_gas/** | Gaussian .com for opt+freq, gas phase. |
| **QCALC_toluene/** | Gaussian .com for opt+freq, SMD/toluene. |
| **SP_gas/** | Single-point .com (after `--qprep-sp` or `DEFAULT_RUN_QPREP_SP = True`), gas. |
| **SP_toluene/** | Single-point .com, toluene. |

QCORR writes into the same `QCALC_<solvent>/` tree (e.g. `success/`, `failed/`, fixed inputs). The pipeline does not run Gaussian or GoodVibes; you run those yourself after the .com (and later SP .com) are generated.

---

## Requirements

- **Python:** `pandas`, `rdkit`, and **aqme** (`pip install aqme`).
- **CREST** (for CSEARCH) and **Gaussian** (for the QM jobs) on your `PATH` if you use the full pipeline.

---

## Summary

- **Where:** Everything runs from **deepchem/aqme/**. Reads **deepchem/data_mingap.csv**; writes **aqme_input.csv**, **CSEARCH/**, **QCALC_***, **SP_*** inside **aqme/**.
- **What:** Convert data_mingap.csv to AQME input, then generate Gaussian inputs (and optionally run QCORR) for gas or SMD/toluene at M06-2X/def2-TZVP and M06-2X/6-31G(d).
- **Why:** So you can recompute Diels–Alder data in a consistent solvent (or gas) without touching the original CSV.
- **How:** Edit the defaults in the scripts and **Run** them from the **aqme** folder, or use the terminal with `--solvent`, `--qcorr`, `--qprep-sp`, etc. Run Gaussian (and GoodVibes) yourself after the pipeline has written the .com files.
