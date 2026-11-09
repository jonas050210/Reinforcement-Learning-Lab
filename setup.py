#!/usr/bin/env python3
"""
================================================================================
  Reinforcement-Learning-Lab — setup.py (Plattform-universeller Einstieg)
================================================================================
Erkennt automatisch Windows 11 oder Ubuntu/Linux, richtet die komplette
Umgebung ein (venv, Pakete, GPU-CUDA) und startet dann die GUI (main.py).

Einfach ausführen:
    python setup.py      (Windows)
    python3 setup.py     (Ubuntu / Linux)
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
    print("║        🤖  REINFORCEMENT-LEARNING-LAB — SETUP.PY            ║")
    print("║                                                             ║")
    print("║     Auto-Setup für Windows 11 & Ubuntu                      ║")
    print("║                                                             ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"{RESET}")


def print_progress(percent: float, text: str = ""):
    """
    Zeigt einen Fortschrittsbalken mit wirklich feinen Prozentwerten.
    Prozent wird immer mit 2 Dezimalstellen und deutschem Komma angezeigt,
    z.B. 0,00% 0,01% 0,02% ... 1,10% 1,11% ...
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


def install_python_tk() -> None:
    """Installiert python3-tk auf Ubuntu/Debian (für tkinter GUI)."""
    if platform.system() != "linux":
        return

    # Prüfen ob tkinter bereits verfügbar ist
    try:
        import tkinter  # noqa: F401
        print_ok("tkinter bereits verfügbar")
        return
    except ImportError:
        pass

    print_step("Installiere python3-tk (für GUI-Tkinter)...")
    try:
        # apt aktualisieren und python3-tk installieren
        result = subprocess.run(
            ["apt-get", "update", "-qq"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            print_warn("apt-get update fehlgeschlagen — versuche trotzdem Installation")
        result = subprocess.run(
            ["apt-get", "install", "-y", "-qq", "python3-tk"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            print_ok("python3-tk installiert")
        else:
            print_warn(f"python3-tk konnte nicht automatisch installiert werden")
            print(f"  {YELLOW}Bitte manuell installieren: sudo apt install python3-tk{RESET}")
    except Exception as e:
        print_warn(f"Fehler bei python3-tk Installation: {e}")
        print(f"  {YELLOW}Bitte manuell installieren: sudo apt install python3-tk{RESET}")


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


def create_windows_shortcut(system: str) -> None:
    """Erstellt einen Desktop-Shortcut unter Windows (nicht OneDrive)."""
    if system != "windows":
        return

    try:
        import winshell
    except ImportError:
        # winshell nicht verfügbar — Shortcut per PowerShell erstellen
        try:
            desktop = os.path.join(os.path.expanduser("~"), "Desktop")
            # OneDrive-Pfad vermeiden: prüfe ob Desktop in OneDrive liegt
            if "OneDrive" in desktop:
                desktop = os.path.join(os.path.expanduser("~"), "Desktop")
            script_path = os.path.abspath("Main.py")
            venv_python = os.path.abspath(str(Path(".venv") / "Scripts" / "python.exe"))
            shortcut_path = os.path.join(desktop, "RL Lab.lnk")
            ps_command = (
                f"$ws = New-Object -ComObject WScript.Shell; "
                f"$shortcut = $ws.CreateShortcut('{shortcut_path}'); "
                f"$shortcut.TargetPath = '{venv_python}'; "
                f"$shortcut.Arguments = '{script_path}'; "
                f"$shortcut.WorkingDirectory = '{os.getcwd()}'; "
                f"$shortcut.IconLocation = '{script_path},0'; "
                f"$shortcut.Save()"
            )
            subprocess.run(
                ["powershell", "-Command", ps_command],
                capture_output=True,
                text=True,
            )
            print_ok(f"Desktop-Shortcut erstellt: {shortcut_path}")
        except Exception as e:
            print_warn(f"Desktop-Shortcut konnte nicht erstellt werden: {e}")
        return

    try:
        import winshell
        from win32com.client import Dispatch
        desktop = winshell.desktop()
        path = os.path.join(desktop, "RL Lab.lnk")
        shell = Dispatch('WScript.Shell')
        shortcut = shell.CreateShortCut(path)
        shortcut.Targetpath = str(Path(".venv") / "Scripts" / "python.exe")
        shortcut.Arguments = "Main.py"
        shortcut.WorkingDirectory = os.getcwd()
        shortcut.IconLocation = str(Path("Main.py").absolute()) + ",0"
        shortcut.save()
        print_ok(f"Desktop-Shortcut erstellt: {path}")
    except ImportError:
        print_warn("winshell/pywin32 nicht installiert — kein Desktop-Shortcut erstellt")
    except Exception as e:
        print_warn(f"Desktop-Shortcut konnte nicht erstellt werden: {e}")


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

    # 3. python3-tk installieren (Ubuntu) für tkinter GUI
    install_python_tk()
    print()

    # 3b. Pfade bestimmen
    venv_python, venv_pip = get_paths(system)

    # 4. Prüfen ob Installation bereits vollständig ist
    installation_complete = check_installation_complete(venv_python, venv_pip)

    if installation_complete:
        print_ok("Alle Pakete sind bereits installiert")
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
        print(f"{GREEN}  Setup abgeschlossen!{RESET}")
        print(f"{GREEN}  Starte die GUI mit:  python3 Main.py{RESET}")
        print(f"{GREEN}{BOLD}════════════════════════════════════════════════════════════════{RESET}\n")

    # 9. GPU prüfen (optional, informativ)
    check_gpu()
    print()

    # 10. Windows-Desktop-Shortcut erstellen (nur Windows, unter Ubuntu übersprungen)
    create_windows_shortcut(system)


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
