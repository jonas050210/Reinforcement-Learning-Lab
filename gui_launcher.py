#!/usr/bin/env python3
"""
GUI-Launcher für Reinforcement-Learning-Lab auf Windows 11 & Ubuntu.

Öffnet ein echtes GUI-Fenster (Tkinter) mit Buttons für:
- Training starten
- Live-Demo ansehen
- Modell auswerten

Kein Terminal nötig — einfach python Main.py oder doppelklick.
"""

import subprocess
import sys
import os
import platform
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Plattform-universeller Pfad zum venv-Python
if platform.system() == "Windows":
    VENV_PYTHON = os.path.join(SCRIPT_DIR, ".venv", "Scripts", "python.exe")
else:
    VENV_PYTHON = os.path.join(SCRIPT_DIR, ".venv", "bin", "python")


class RLLauncher(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Reinforcement-Learning-Lab")
        self.geometry("700x550")
        self.configure(bg="#2c3e50")

        # Stil
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TFrame", background="#2c3e50")
        style.configure(
            "TLabel",
            background="#2c3e50",
            foreground="#ecf0f1",
            font=("Segoe UI" if platform.system() == "Windows" else "Arial", 11),
        )
        style.configure(
            "Title.TLabel",
            font=("Segoe UI" if platform.system() == "Windows" else "Arial", 18, "bold"),
            foreground="#3498db",
        )
        style.configure(
            "TButton",
            font=("Segoe UI" if platform.system() == "Windows" else "Arial", 11, "bold"),
            padding=12,
        )

        self._build_ui()

    def _build_ui(self):
        # Hauptcontainer
        main = ttk.Frame(self, padding="30")
        main.pack(fill=tk.BOTH, expand=True)

        # Titel
        ttk.Label(
            main, text="🤖 Reinforcement-Learning-Lab", style="Title.TLabel"
        ).pack(pady=(0, 10))
        ttk.Label(
            main,
            text="3D-Roboter RL auf Windows & Ubuntu mit GPU-Unterstützung",
        ).pack(pady=(0, 30))

        # Prüfe ob venv existiert
        self.venv_exists = os.path.exists(VENV_PYTHON)

        if not self.venv_exists:
            ttk.Label(
                main,
                text="⚠️ Virtuelle Umgebung noch nicht eingerichtet!",
                foreground="#e74c3c",
                font=("Segoe UI" if platform.system() == "Windows" else "Arial", 10, "bold"),
            ).pack(pady=(0, 10))
            ttk.Label(
                main,
                text="Bitte zuerst Main.py ausführen zum Installieren.",
                foreground="#bdc3c7",
            ).pack(pady=(0, 20))

        # Buttons
        btn_frame = ttk.Frame(main)
        btn_frame.pack(fill=tk.BOTH, expand=True, pady=20)

        buttons = [
            (
                "🚀 Training starten",
                self.start_training,
                "Startet PPO-Training (500k Timesteps)",
            ),
            (
                "🎮 Live-Demo öffnen",
                self.start_demo,
                "Öffnet MuJoCo-Fenster mit trainiertem Arm",
            ),
            (
                "📊 Modell auswerten",
                self.start_eval,
                "50 Test-Episoden mit Statistiken",
            ),
            (
                "❌ Beenden",
                self.quit,
                "Programm schließen",
            ),
        ]

        for text, cmd, desc in buttons:
            frame = ttk.Frame(btn_frame)
            frame.pack(fill=tk.X, pady=8)
            btn = ttk.Button(
                frame,
                text=text,
                command=cmd,
                width=30,
                state="normal" if self.venv_exists else "disabled",
            )
            btn.pack(side=tk.LEFT, padx=(0, 10))
            ttk.Label(
                frame, text=desc, foreground="#bdc3c7"
            ).pack(side=tk.LEFT)

        # Log-Bereich
        ttk.Label(main, text="Ausgabe:").pack(anchor=tk.W, pady=(20, 5))
        self.log = scrolledtext.ScrolledText(
            main,
            height=10,
            bg="#34495e",
            fg="#ecf0f1",
            insertbackground="#ecf0f1",
            font=("Consolas" if platform.system() == "Windows" else "Courier", 9),
        )
        self.log.pack(fill=tk.BOTH, expand=True)

        # Status
        self.status = ttk.Label(
            main,
            text="Bereit" if self.venv_exists else "Setup erforderlich",
            foreground="#2ecc71" if self.venv_exists else "#e74c3c",
        )
        self.status.pack(anchor=tk.W, pady=(10, 0))

    def _log(self, text: str):
        self.log.insert(tk.END, text + "\n")
        self.log.see(tk.END)
        self.update_idletasks()

    def _run_script(self, script_name: str, label: str):
        """Führt ein Python-Skript im venv aus und zeigt die Ausgabe im Log."""
        if not self.venv_exists:
            messagebox.showerror(
                "Fehler",
                "Virtuelle Umgebung nicht gefunden!\n"
                "Bitte zuerst Main.py ausführen zum Installieren.",
            )
            return

        script_path = os.path.join(SCRIPT_DIR, script_name)
        if not os.path.exists(script_path):
            messagebox.showerror(
                "Fehler",
                f"Skript nicht gefunden: {script_name}\n"
                f"Erwartet unter: {SCRIPT_DIR}",
            )
            return

        self.status.config(text=f"{label} läuft...", foreground="#f39c12")
        self._log(f">>> Starte {label}...")
        self._log(f">>> Python: {VENV_PYTHON}")
        self._log(f">>> Skript: {script_path}")
        self._log("-" * 50)

        try:
            proc = subprocess.Popen(
                [VENV_PYTHON, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=SCRIPT_DIR,
                bufsize=1,
            )

            # Zeile für Zeile auslesen (live Ausgabe)
            for line in proc.stdout:
                self._log(line.rstrip())

            proc.wait()
            if proc.returncode == 0:
                self.status.config(
                    text=f"{label} abgeschlossen ✅", foreground="#2ecc71"
                )
                self._log("-" * 50)
                self._log(f">>> {label} erfolgreich beendet.\n")
            else:
                self.status.config(
                    text=f"{label} Fehler ❌", foreground="#e74c3c"
                )
                self._log("-" * 50)
                self._log(f">>> {label} mit Fehlercode {proc.returncode} beendet.\n")

        except Exception as e:
            self.status.config(text="Fehler ❌", foreground="#e74c3c")
            self._log(f">>> Fehler: {e}\n")
            messagebox.showerror("Fehler", str(e))

    def start_training(self):
        self._run_script("train.py", "Training")

    def start_demo(self):
        self._run_script("demo.py", "Demo")

    def start_eval(self):
        self._run_script("eval.py", "Evaluation")


if __name__ == "__main__":
    app = RLLauncher()
    app.mainloop()
