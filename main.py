#!/usr/bin/env python3
"""
================================================================================
  Reinforcement-Learning-Lab — main.py (Wrapper)
================================================================================
Wrapper für Main.py (großes M). Unter Linux/Ubuntu ist die Dateinamen-Groß-
und Kleinschreibung case-sensitive.

Einfach ausführen:
    python3 main.py
    oder
    python main.py
================================================================================
"""

import os
import sys
import subprocess
from pathlib import Path


def main():
    # Pfad zu Main.py im selben Verzeichnis
    main_py = Path(__file__).parent / "Main.py"

    if not main_py.exists():
        print("FEHLER: Main.py nicht gefunden!")
        print(f"Erwartet unter: {main_py}")
        sys.exit(1)

    # Aktuelles Arbeitsverzeichnis beibehalten (für relative Pfade wie .venv, requirements.txt)
    cwd = Path(__file__).parent.resolve()

    # venv-Python bevorzugen, falls vorhanden
    venv_python = cwd / ".venv" / "bin" / "python"
    if venv_python.exists():
        python_cmd = str(venv_python)
    else:
        python_cmd = sys.executable

    print(f"Starte Main.py mit {python_cmd} ...")
    print(f"Arbeitsverzeichnis: {cwd}")
    print()

    try:
        subprocess.run([python_cmd, str(main_py)], cwd=cwd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nMain.py wurde mit Fehlercode {e.returncode} beendet")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nAbbruch durch Benutzer")
        sys.exit(0)


if __name__ == "__main__":
    main()
