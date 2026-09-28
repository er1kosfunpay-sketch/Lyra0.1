"""Run the full test suite:  python tests/run_all.py  (exit code = pytest exit code)."""
import subprocess,sys
from pathlib import Path

def main():
 root=Path(__file__).resolve().parents[1]
 r=subprocess.run([sys.executable,'-m','pytest','tests/','-q'],cwd=root)
 print(f'SMOKE {"PASS" if r.returncode==0 else "FAIL"} (exit={r.returncode})')
 return r.returncode

if __name__=='__main__':raise SystemExit(main())
