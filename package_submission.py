"""Assemble the course ZIP: submission/construction.zip containing folder "2 - construction/".

Copies code, notebooks (with outputs), notes, the shipped split and result files, and
anime.csv (its public mirror can't be byte-verified, so the exact file ships). rating.csv is
NOT copied (stable link, SHA-1 verified -> README.md). Models and evolution/*.ttl are not
copied (regenerable, large).

Run:  python package_submission.py
"""

import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "submission"
DEST = OUT / "2 - construction"

INCLUDE = [
    "README.md",
    "requirements.txt",
    "_context/glossary.md",
    "Apply/notes.md",
    "Apply/data/readme.md",
    "Apply/data/anime.csv",
    "Apply/*.ipynb",
    "Apply/src/*.py",
    "Apply/src/*.ttl",
    "Apply/src/web/*",
    "docs/*",
    "docs/data/*",
    "Apply/data/generated/provenance.json",
    "Apply/data/generated/split/*",
    "Apply/data/generated/results/*",
    "Learn/0[1-4]_*/*.ipynb",
    "Learn/0[1-4]_*/*.py",
    "Learn/0[1-4]_*/*.ttl",
    "Learn/0[1-4]_*/notes.md",
]

# repo path -> path inside "2 - construction" (when it differs)
RELOCATE = {"for_portfolio.md": "_context/for_portfolio.md"}  # decisions ledger sits with the glossary


def main() -> None:
    if DEST.exists():
        shutil.rmtree(DEST)
    # skip near-empty placeholder stubs (e.g. unused scratch.py)
    files = sorted({p for pat in INCLUDE for p in ROOT.glob(pat) if p.is_file() and p.stat().st_size > 100})
    files += [ROOT / f for f in RELOCATE]
    for src in files:
        rel = str(src.relative_to(ROOT))
        dst = DEST / RELOCATE.get(rel, rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    zpath = OUT / "construction.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(DEST.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(OUT))
    print(f"{len(files)} files -> {DEST.relative_to(ROOT)}  |  zip {zpath.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
