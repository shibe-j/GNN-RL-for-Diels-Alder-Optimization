"""
AQME pipeline for data_mingap-derived species.

Script lives in deepchem/aqme/. Reads data_mingap.csv from deepchem/ (parent)
and aqme_input.csv from this folder. Writes CSEARCH/, QCALC_*, SP_* into this folder.

Level of theory:
  - Opt + freq: M06-2X/def2-TZVP  [gas or SMD/toluene]
  - Single-point: M06-2X/6-31G(d) [same environment for SPC with GoodVibes]

Click Run to use the defaults below (no terminal args). Or run from terminal with
  python run_aqme_pipeline.py --solvent toluene --all
etc.
"""

import argparse
import os
import sys
from pathlib import Path

# Script lives in deepchem/aqme/; dataset is in deepchem/
AQME_DIR = Path(__file__).resolve().parent
DEEPCHEM_DIR = AQME_DIR.parent
AQME_INPUT_CSV = AQME_DIR / "aqme_input.csv"
DATA_MINGAP = DEEPCHEM_DIR / "data_mingap.csv"

# -----------------------------------------------------------------------------
# Default usage (click Run) — edit these instead of passing terminal args
# -----------------------------------------------------------------------------
DEFAULT_SOLVENT = "gas"           # "gas" or "toluene"
DEFAULT_RUN_CONVERTER = True     # ensure aqme_input.csv exists from data_mingap.csv
DEFAULT_RUN_CSEARCH = True       # conformational sampling
DEFAULT_RUN_QPREP_OPT = True     # write Gaussian .com for opt+freq
DEFAULT_RUN_QCORR = False        # set True after you have run Gaussian on QCALC_*
DEFAULT_RUN_QPREP_SP = False     # set True after QCORR to prepare single-point inputs
DEFAULT_NPROCS = 4
DEFAULT_MEM = "4GB"
DEFAULT_WORKDIR = AQME_DIR       # CSEARCH, QCALC_*, SP_* created here
# -----------------------------------------------------------------------------

# Level of theory
OPT_FREQ_BASIS = "def2TZVP"
SP_BASIS = "6-31G(d)"
METHOD = "M062X"

# Route lines (Gaussian)
def _qm_input_opt_freq(ts: bool, solvent: str) -> str:
    base = f"{METHOD}/{OPT_FREQ_BASIS}"
    if solvent == "toluene":
        base += " scrf=(smd,solvent=toluene)"
    if ts:
        return base + " opt=(ts,calcfc,noeigen,maxstep=5) freq=noraman"
    return base + " opt freq=noraman"


def _qm_input_sp(solvent: str) -> str:
    base = f"{METHOD}/{SP_BASIS}"
    if solvent == "toluene":
        base += " scrf=(smd,solvent=toluene)"
    return base


def _ensure_aqme_input():
    if AQME_INPUT_CSV.exists():
        return
    if not DATA_MINGAP.exists():
        raise FileNotFoundError(
            f"Neither {AQME_INPUT_CSV} nor {DATA_MINGAP} found. Run aqme_converter.py first or ensure data_mingap.csv exists in deepchem/."
        )
    # Run converter
    from aqme_converter import build_aqme_input_csv
    build_aqme_input_csv(DATA_MINGAP, AQME_INPUT_CSV, include_reactants=True)
    print(f"Created {AQME_INPUT_CSV}")


def run_csearch(w_dir: Path, nprocs: int = 8):
    try:
        from aqme.csearch import csearch
    except ImportError:
        print("AQME not installed. Install with: pip install aqme", file=sys.stderr)
        sys.exit(1)
    os.chdir(w_dir)
    csearch(
        input=str(AQME_INPUT_CSV),
        program="crest",
        nprocs=nprocs,
        cregen=True,
        cregen_keywords="--ethr 0.1 --rthr 0.2 --bthr 0.3 --ewin 1",
    )


def run_qprep_opt_freq(w_dir: Path, solvent: str, mem: str = "4GB", nprocs: int = 4):
    try:
        from aqme.qprep import qprep
    except ImportError:
        print("AQME not installed. Install with: pip install aqme", file=sys.stderr)
        sys.exit(1)
    os.chdir(w_dir)
    csearch_dir = Path("CSEARCH")
    sdf_ts = [str(p) for p in csearch_dir.glob("TS*crest.sdf")]
    sdf_other = [str(p) for p in csearch_dir.glob("Diene_*.sdf")] + [str(p) for p in csearch_dir.glob("Dienophile_*.sdf")]
    dest = f"QCALC_{solvent}"
    Path(dest).mkdir(parents=True, exist_ok=True)
    if sdf_ts:
        qprep(
            files=sdf_ts,
            program="gaussian",
            qm_input=_qm_input_opt_freq(ts=True, solvent=solvent),
            mem=mem,
            nprocs=nprocs,
            destination=str(Path(w_dir) / dest),
        )
    if sdf_other:
        qprep(
            files=sdf_other,
            program="gaussian",
            qm_input=_qm_input_opt_freq(ts=False, solvent=solvent),
            mem=mem,
            nprocs=nprocs,
            destination=str(Path(w_dir) / dest),
        )
    print(f"QPREP opt+freq done. Inputs in {w_dir / dest}. Run Gaussian, then re-run this script with --qcorr and --solvent {solvent}.")


def run_qcorr(w_dir: Path, solvent: str, mem: str = "4GB", nprocs: int = 4):
    try:
        from aqme.qcorr import qcorr
    except ImportError:
        print("AQME not installed. Install with: pip install aqme", file=sys.stderr)
        sys.exit(1)
    os.chdir(w_dir)
    qcalc = Path(f"QCALC_{solvent}")
    logs = list(qcalc.glob("*.log"))
    if not logs:
        logs = list(qcalc.rglob("*.log"))
    if not logs:
        print(f"No LOG files in {qcalc}. Run Gaussian first.", file=sys.stderr)
        return
    qcorr(
        files=str(qcalc / "*.log"),
        freq_conv="opt=(calcfc,maxstep=5)",
        mem=mem,
        nprocs=nprocs,
    )
    print("QCORR done. Check QCALC_*/success/ and failed/.")


def run_qprep_sp(w_dir: Path, solvent: str, mem: str = "4GB", nprocs: int = 4):
    try:
        from aqme.qprep import qprep
    except ImportError:
        print("AQME not installed. Install with: pip install aqme", file=sys.stderr)
        sys.exit(1)
    os.chdir(w_dir)
    success_dir = Path(f"QCALC_{solvent}") / "success"
    log_files = list(success_dir.glob("*.log"))
    if not log_files:
        print(f"No LOG files in {success_dir}. Run Gaussian and QCORR first.", file=sys.stderr)
        return
    dest = f"SP_{solvent}"
    Path(dest).mkdir(parents=True, exist_ok=True)
    qprep(
        files=[str(p) for p in log_files],
        program="gaussian",
        qm_input=_qm_input_sp(solvent),
        mem=mem,
        nprocs=nprocs,
        destination=str(Path(w_dir) / dest),
        suffix="SP",
    )
    print(f"QPREP single-point done. Inputs in {w_dir / dest}. Run Gaussian for SPC, then use GoodVibes with these LOGs + SP LOGs.")


def main():
    parser = argparse.ArgumentParser(description="AQME pipeline: M06-2X/def2-TZVP // M06-2X/6-31G(d), gas or SMD/toluene")
    parser.add_argument("--solvent", choices=("gas", "toluene"), default="gas",
                        help="Gas phase or SMD/toluene")
    parser.add_argument("--csearch", action="store_true", help="Run CSEARCH only")
    parser.add_argument("--qprep-opt", action="store_true", help="Run QPREP for opt+freq only")
    parser.add_argument("--qcorr", action="store_true", help="Run QCORR on LOGs")
    parser.add_argument("--qprep-sp", action="store_true", help="Run QPREP for single-point (on success LOGs)")
    parser.add_argument("--all", action="store_true", help="Run CSEARCH + QPREP opt+freq (full prep for Gaussian)")
    parser.add_argument("--nprocs", type=int, default=4)
    parser.add_argument("--mem", type=str, default="4GB")
    parser.add_argument("--workdir", type=Path, default=AQME_DIR, help="Working directory (default: this folder)")
    args = parser.parse_args()

    w_dir = args.workdir.resolve()
    w_dir.mkdir(parents=True, exist_ok=True)
    _ensure_aqme_input()

    if args.csearch or args.all:
        run_csearch(w_dir, nprocs=args.nprocs)
    if args.qprep_opt or args.all:
        run_qprep_opt_freq(w_dir, args.solvent, mem=args.mem, nprocs=args.nprocs)
    if args.qcorr:
        run_qcorr(w_dir, args.solvent, mem=args.mem, nprocs=args.nprocs)
    if args.qprep_sp:
        run_qprep_sp(w_dir, args.solvent, mem=args.mem, nprocs=args.nprocs)

    if not any([args.csearch, args.qprep_opt, args.qcorr, args.qprep_sp, args.all]):
        # Default: run full prep (csearch + qprep opt)
        run_csearch(w_dir, nprocs=args.nprocs)
        run_qprep_opt_freq(w_dir, args.solvent, mem=args.mem, nprocs=args.nprocs)
        print("\nNext steps:")
        print(f"  1. Run Gaussian on all .com files in {w_dir / ('QCALC_' + args.solvent)}")
        print(f"  2. Set DEFAULT_RUN_QCORR = True and Run, or: --solvent {args.solvent} --qcorr")
        print(f"  3. Set DEFAULT_RUN_QPREP_SP = True and Run, or: --solvent {args.solvent} --qprep-sp")
        print("  4. Run Gaussian on the generated SP .com files, then GoodVibes for thermochemistry.")


def run_with_defaults():
    """Use in-file defaults (no terminal args). Call this when you click Run."""
    w_dir = DEFAULT_WORKDIR.resolve()
    w_dir.mkdir(parents=True, exist_ok=True)
    if DEFAULT_RUN_CONVERTER:
        _ensure_aqme_input()
    if DEFAULT_RUN_CSEARCH:
        run_csearch(w_dir, nprocs=DEFAULT_NPROCS)
    if DEFAULT_RUN_QPREP_OPT:
        run_qprep_opt_freq(w_dir, DEFAULT_SOLVENT, mem=DEFAULT_MEM, nprocs=DEFAULT_NPROCS)
    if DEFAULT_RUN_QCORR:
        run_qcorr(w_dir, DEFAULT_SOLVENT, mem=DEFAULT_MEM, nprocs=DEFAULT_NPROCS)
    if DEFAULT_RUN_QPREP_SP:
        run_qprep_sp(w_dir, DEFAULT_SOLVENT, mem=DEFAULT_MEM, nprocs=DEFAULT_NPROCS)
    if not any([DEFAULT_RUN_CSEARCH, DEFAULT_RUN_QPREP_OPT, DEFAULT_RUN_QCORR, DEFAULT_RUN_QPREP_SP]):
        print("All DEFAULT_RUN_* are False. Set DEFAULT_RUN_CSEARCH and DEFAULT_RUN_QPREP_OPT to True to generate inputs.")
    else:
        print(f"\nDone (solvent={DEFAULT_SOLVENT}). Next: run Gaussian on QCALC_{DEFAULT_SOLVENT}/ then set DEFAULT_RUN_QCORR / DEFAULT_RUN_QPREP_SP and Run again.")


if __name__ == "__main__":
    # If no CLI args, use in-file defaults (click Run)
    if len(sys.argv) == 1:
        run_with_defaults()
    else:
        main()
