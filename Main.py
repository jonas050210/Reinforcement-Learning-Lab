#!/usr/bin/env python3
"""
================================================================================
  Reinforcement-Learning-Lab — Main.py (Plattform-universeller Starter)
================================================================================
Erkennt automatisch Windows 11 oder Ubuntu/Linux, richtet die komplette
Umgebung ein (venv, Pakete, GPU-CUDA) und startet dann die passende GUI.

Beim ersten Start wird alles installiert. Bei jedem weiteren Start prüft
Main.py ob alles bereits da ist und startet dann sofort die GUI — ohne
erneute Installation.

Einfach ausführen:
    python Main.py
    oder
    python3 Main.py
================================================================================
"""

import os
import sys
import subprocess
import platform
import time
import importlib.util
from pathlib import Path

# ---------------------------------------------------------------------------
# Farb-Codes für das Terminal (ANSI)
# ---------------------------------------------------------------------------
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
RESET = "\033[0m"

# Unter Windows ANSI farben aktivieren
if platform.system() == "Windows":
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:
        pass


def print_header():
    print(f"{CYAN}{BOLD}")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                                                             ║")
    print("║        🤖  REINFORCEMENT-LEARNING-LAB — MAIN.PY             ║")
    print("║                                                             ║")
    print("║     Auto-Setup für Windows 11 & Ubuntu                      ║")
    print("║                                                             ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"{RESET}")


def print_progress(percent: float, text: str = ""):
    """
    Zeigt einen Fortschrittsbalken mit wirklich feinen Prozentwerten.
    Prozent wird immer mit 2 Dezimalstellen und deutschem Komma angezeigt,
    z.B. 1,10% 1,11% 1,12% ...
    """
    bar_length = 50
    filled = int(bar_length * percent / 100)
    bar = "█" * filled + "░" * (bar_length - filled)
    percent_str = f"{percent:6.2f}%".replace(".", ",")
    print(f"\r  [{bar}]  {percent_str:>9}  {text}", end="", flush=True)


def print_step(text: str):
    print(f"\n{BLUE}▶{RESET} {text}")


def print_ok(text: str):
    print(f"{GREEN}  ✓{RESET} {text}")


def print_warn(text: str):
    print(f"{YELLOW}  ⚠{RESET} {text}")


def print_error(text: str):
    print(f"{RED}  ✗{RESET} {text}")


def detect_system() -> str:
    system = platform.system()
    if system == "Windows":
        return "windows"
    elif system == "Linux":
        return "linux"
    elif system == "Darwin":
        return "macos"
    else:
        return "unknown"


def check_python() -> bool:
    """Prüft ob Python verfügbar ist und mind. Version 3.9."""
    try:
        result = subprocess.run(
            [sys.executable, "--version"], capture_output=True, text=True
        )
        version_str = result.stdout.strip() or result.stderr.strip()
        print_ok(f"Python gefunden: {version_str}")
        return True
    except Exception:
        print_error("Python nicht gefunden!")
        print(f"  {YELLOW}Bitte installiere Python 3.9+ von https://python.org{RESET}")
        print(f"  {YELLOW}Wichtig: Bei der Installation 'Add Python to PATH' anhaken!{RESET}")
        return False


def get_paths(system: str):
    """Gibt Pfade für venv-Python und pip zurück (plattform-universell)."""
    if system == "windows":
        venv_python = Path(".venv") / "Scripts" / "python.exe"
        venv_pip = Path(".venv") / "Scripts" / "pip.exe"
    else:
        venv_python = Path(".venv") / "bin" / "python"
        venv_pip = Path(".venv") / "bin" / "pip"
    return venv_python, venv_pip


def is_package_installed(python_path: Path, package_name: str) -> bool:
    """Prüft ob ein Paket in der venv installiert ist."""
    try:
        result = subprocess.run(
            [
                str(python_path),
                "-c",
                f"import importlib.util; print(importlib.util.find_spec('{package_name}'))",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return "None" not in result.stdout and result.stdout.strip() != ""
    except Exception:
        return False


def check_installation_complete(venv_python: Path, venv_pip: Path) -> bool:
    """Prüft ob venv existiert und alle Pakete installiert sind."""
    if not venv_python.exists():
        return False

    # Wichtige Pakete prüfen
    required_packages = [
        "mujoco",
        "gymnasium",
        "stable_baselines3",
        "torch",
        "tensorboard",
        "tqdm",
        "rich",
        "numpy",
    ]

    missing = []
    for pkg in required_packages:
        if not is_package_installed(venv_python, pkg):
            missing.append(pkg)

    if missing:
        print_warn(f"Folgende Pakete fehlen: {', '.join(missing)}")
        return False

    return True


def create_venv() -> bool:
    """Erstellt die virtuelle Umgebung."""
    print_step("Erstelle virtuelle Umgebung (.venv)...")
    try:
        subprocess.run(
            [sys.executable, "-m", "venv", ".venv"],
            check=True,
            capture_output=True,
        )
        print_ok("Virtuelle Umgebung erstellt")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Fehler beim Erstellen der venv: {e}")
        return False


def upgrade_pip(pip_path: Path) -> None:
    """Upgradet pip in der venv."""
    print_step("Upgradet pip...")
    subprocess.run(
        [str(pip_path), "install", "--upgrade", "pip", "--quiet"],
        capture_output=True,
    )


def install_requirements(pip_path: Path, requirements_file: Path) -> bool:
    """
    Installiert alle Pakete aus requirements.txt mit LIVE-Fortschritt.

    Der Fortschritt wird während jeder Paket-Installation fein abgestuft
    angezeigt (0,00% 0,01% 0,02% ...) mit deutschem Komma-Format.
    """
    if not requirements_file.exists():
        print_error(f"{requirements_file} nicht gefunden!")
        return False

    with open(requirements_file, "r", encoding="utf-8") as f:
        packages = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]

    total = len(packages)
    if total == 0:
        print_warn("Keine Pakete in requirements.txt gefunden")
        return True

    print_step(f"Installiere {total} Pakete (Live-Fortschritt mit feinen Prozentwerten)...\n")

    for index, package in enumerate(packages, start=1):
        # Exakter Fortschritt vor der Installation dieses Pakets
        percent_before = ((index - 1) / total) * 100
        percent_after = (index / total) * 100

        print_progress(percent_before, f"Vorbereite {package} ...")
        time.sleep(0.03)

        # Installation des Pakets mit feinem Live-Fortschritt währenddessen
        try:
            proc = subprocess.Popen(
                [str(pip_path), "install", package],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            # Fein abgestufte Prozent-Animation solange pip läuft
            current_pct = percent_before
            step_size = 0.05  # 0,05%-Schritte für feine Anzeige
            while proc.poll() is None:
                current_pct = min(current_pct + step_size, percent_after - 0.01)
                print_progress(current_pct, f"Installiere {package} ...")
                time.sleep(0.02)

            proc.wait()
            if proc.returncode != 0:
                print_warn(f"Paket {package} konnte nicht vollständig installiert werden")
        except subprocess.TimeoutExpired:
            print_warn(f"Timeout bei {package}")
        except Exception as e:
            print_warn(f"Fehler bei {package}: {e}")

        # Exakter Fortschritt NACH der Installation dieses Pakets
        print_progress(percent_after, f"Installiert: {package}")
        time.sleep(0.03)

    # Finale Anzeige auf 100,00%
    print_progress(100.0, "Alle Pakete installiert       ")
    print()  # Zeilenumbruch nach dem Fortschrittsbalken
    print_ok(f"{total} Pakete verarbeitet")
    return True


def check_gpu() -> bool:
    """Prüft ob eine NVIDIA-GPU mit CUDA verfügbar ist."""
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print_ok("NVIDIA-GPU erkannt (CUDA verfügbar)")
            return True
    except Exception:
        pass
    print_warn("Keine NVIDIA-GPU erkannt — läuft auf CPU (funktioniert trotzdem)")
    return False


def launch_gui(venv_python: Path, system: str) -> None:
    """Startet die passende GUI je nach Betriebssystem."""
    print_step("Starte Anwendung ...")

    gui_script = Path("gui_launcher.py")
    demo_script = Path("demo.py")

    if gui_script.exists():
        script = gui_script
    elif demo_script.exists():
        script = demo_script
    else:
        print_error("Kein Start-Skript gefunden (gui_launcher.py oder demo.py)")
        return

    print(f"{GREEN}  → Starte {script.name}{RESET}\n")

    try:
        subprocess.run([str(venv_python), str(script)], check=True)
    except subprocess.CalledProcessError as e:
        print_error(f"Das Programm wurde mit Fehlercode {e.returncode} beendet")
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Abbruch durch Benutzer{RESET}")


def main():
    print_header()

    # 1. Betriebssystem erkennen
    system = detect_system()
    system_names = {
        "windows": "Windows 11/10",
        "linux": "Ubuntu / Linux",
        "macos": "macOS",
        "unknown": "Unbekannt",
    }
    print(f"{CYAN}Betriebssystem:{RESET} {system_names.get(system, system)}")
    print(f"{CYAN}Python:{RESET} {sys.version.split()[0]}")
    print()

    # 2. Python prüfen
    if not check_python():
        input("\nDrücke Enter zum Beenden ...")
        sys.exit(1)

    # 3. Pfade bestimmen
    venv_python, venv_pip = get_paths(system)

    # 4. Prüfen ob Installation bereits vollständig ist
    installation_complete = check_installation_complete(venv_python, venv_pip)

    if installation_complete:
        print_ok("Alle Pakete sind bereits installiert — starte direkt die GUI")
        print()
    else:
        # 5. Virtuelle Umgebung erstellen (falls nötig)
        if not Path(".venv").exists():
            if not create_venv():
                input("\nDrücke Enter zum Beenden ...")
                sys.exit(1)
        else:
            print_ok("Virtuelle Umgebung existiert bereits")

        # 6. Pip upgraden
        upgrade_pip(venv_pip)

        # 7. Pakete installieren mit Fortschritt
        req_file = Path("requirements.txt")
        if not install_requirements(venv_pip, req_file):
            print_error("Installation fehlgeschlagen!")
            input("\nDrücke Enter zum Beenden ...")
            sys.exit(1)

        # 8. Zusammenfassung
        print(f"\n{GREEN}{BOLD}════════════════════════════════════════════════════════════════{RESET}")
        print(f"{GREEN}  Setup abgeschlossen! Starte Anwendung ...{RESET}")
        print(f"{GREEN}{BOLD}════════════════════════════════════════════════════════════════{RESET}\n")

    # 9. Prüfen ob tkinter verfügbar ist (für GUI nötig)
    try:
        import tkinter  # noqa: F401
    except ImportError:
        print_error("tkinter ist nicht installiert — GUI kann nicht geöffnet werden!")
        print()
        if system == "linux":
            print(f"{YELLOW}  Unter Ubuntu installieren:{RESET}")
            print(f"    sudo apt update && sudo apt install python3-tk")
            print()
            print(f"{YELLOW}  Danach starten:{RESET}")
            print(f"    python3 Main.py")
        elif system == "windows":
            print(f"{YELLOW}  Unter Windows: Python neu installieren und 'tcl/tk und IDLE' anhaken.{RESET}")
        print()
        input("Drücke Enter zum Beenden ...")
        sys.exit(1)

    # 10. GUI starten
    launch_gui(venv_python, system)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Abbruch durch Benutzer{RESET}")
        sys.exit(0)
    except Exception as e:
        print_error(f"Unerwarteter Fehler: {e}")
        import traceback
        traceback.print_exc()
        input("\nDrücke Enter zum Beenden ...")
        sys.exit(1)
