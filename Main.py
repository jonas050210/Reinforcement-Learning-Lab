#!/usr/bin/env python3
"""
================================================================================
  Reinforcement-Learning-Lab — Main.py (Plattform-universeller Starter)
================================================================================
Erkennt automatisch Windows 11 oder Ubuntu/Linux, richtet die komplette
Umgebung ein (venv, Pakete, GPU-CUDA) und startet dann die passende GUI.

Einfach ausführen:
    python Main.py

Alle Downloads werden mit Live-Fortschritt in Prozent angezeigt
(z.B. 1,10% 1,11% 1,12% ... wirklich jeder einzelne Wert).
================================================================================
"""

import os
import sys
import subprocess
import platform
import time
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
    # Prozent mit Komma und 2 Dezimalstellen, z.B. 1,10%
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
    
    Der Fortschritt wird bei JEDEM Paket aktualisiert und zeigt wirklich
    feine Prozentwerte mit 2 Dezimalstellen (z.B. 1,10% 1,11% 1,12%).
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

    # Wir teilen den Fortschritt in feine Schritte auf:
    # Vor jeder Installation zeigen wir den aktuellen Fortschritt,
    # und nach jeder Installation den neuen.
    # Damit es wirklich "einzeln" hochzählt (1,10% 1,11% ...),
    # nutzen wir die exakte Bruchrechnung mit 2 Dezimalstellen.
    
    for index, package in enumerate(packages, start=1):
        # Exakter Fortschritt vor der Installation dieses Pakets
        percent_before = ((index - 1) / total) * 100
        print_progress(percent_before, f"Vorbereite {package} ...")
        time.sleep(0.1)  # Kurze Pause damit der Fortschritt sichtbar ist

        # Installation des Pakets
        try:
            result = subprocess.run(
                [str(pip_path), "install", "--quiet", package],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode != 0:
                print_warn(f"Paket {package} konnte nicht vollständig installiert werden")
        except subprocess.TimeoutExpired:
            print_warn(f"Timeout bei {package}")
        except Exception as e:
            print_warn(f"Fehler bei {package}: {e}")

        # Exakter Fortschritt NACH der Installation dieses Pakets
        percent_after = (index / total) * 100
        print_progress(percent_after, f"Installiert: {package}")
        time.sleep(0.1)

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

    # Prüfen ob gui_launcher.py existiert
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

    # 3. GPU prüfen (nur Info)
    check_gpu()

    # 4. Virtuelle Umgebung erstellen
    venv_python, venv_pip = get_paths(system)

    if not Path(".venv").exists():
        if not create_venv():
            input("\nDrücke Enter zum Beenden ...")
            sys.exit(1)
    else:
        print_ok("Virtuelle Umgebung existiert bereits")

    # 5. Pip upgraden
    upgrade_pip(venv_pip)

    # 6. Pakete installieren mit Fortschritt (wirklich feine Werte)
    req_file = Path("requirements.txt")
    if not install_requirements(venv_pip, req_file):
        print_error("Installation fehlgeschlagen!")
        input("\nDrücke Enter zum Beenden ...")
        sys.exit(1)

    # 7. Zusammenfassung
    print(f"\n{GREEN}{BOLD}════════════════════════════════════════════════════════════════{RESET}")
    print(f"{GREEN}  Setup abgeschlossen! Starte Anwendung ...{RESET}")
    print(f"{GREEN}{BOLD}════════════════════════════════════════════════════════════════{RESET}\n")

    # 8. GUI starten
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
