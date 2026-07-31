#!/usr/bin/env python3
from pathlib import Path
import os

def print_paths():
    script_path = Path(__file__).resolve()
    cwd = Path(os.getcwd()).resolve()
    
    print("=" * 60)
    print("📍 PATH CHECKER")
    print("=" * 60)
    print(f"__file__ (script location) : {script_path}")
    print(f"Current working directory  : {cwd}")
    print()
    print("Parent folders:")
    print(f"  .parent                  : {script_path.parent}")
    print(f"  .parent.parent           : {script_path.parent.parent}")
    print(f"  .parent.parent.parent    : {script_path.parent.parent.parent}")
    print()
    print("Looking for 'doc' folder:")
    doc_candidates = [
        script_path.parent / "doc",
        script_path.parent.parent / "doc",
        script_path.parent.parent.parent / "doc",
        cwd / "doc",
    ]
    for i, p in enumerate(doc_candidates):
        exists = "✅ EXISTS" if p.exists() else "❌ missing"
        print(f"  {i+1}. {p}  {exists}")
    print("=" * 60)

if __name__ == "__main__":
    print_paths()