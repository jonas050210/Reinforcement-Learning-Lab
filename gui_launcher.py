#!/usr/bin/env python3
"""
GUI-Launcher für Reinforcement-Learning-Lab auf Windows 11 & Ubuntu.

Öffnet ein echtes GUI-Fenster (Tkinter) mit Buttons für:
- V1: Einfacher Arm (Würfel in Box)
- V2: Arm mit 3 Werkzeugen (Greifer, Sauger, Schaufel)
- V3: Roboterhund lernt laufen

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

        self.title("Reinforcement-Learning-Lab — Alle Versionen")
        self.geometry("800x650")
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
            font=("Segoe UI" if platform.system() == "Windows" else "Arial", 20, "bold"),
            foreground="#3498db",
        )
        style.configure(
            "Subtitle.TLabel",
            font=("Segoe UI" if platform.system() == "Windows" else "Arial", 12),
            foreground="#bdc3c7",
        )
        style.configure(
            "TButton",
            font=("Segoe UI" if platform.system() == "Windows" else "Arial", 11, "bold"),
            padding=14,
        )

        self._build_ui()

    def _build_ui(self):
        # Hauptcontainer mit Scroll
        container = ttk.Frame(self)
        container.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(container, bg="#2c3e50", highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        main = ttk.Frame(scrollable_frame, padding="40")
        main.pack(fill=tk.BOTH, expand=True)

        # Titel
        ttk.Label(main, text="🤖 Reinforcement-Learning-Lab", style="Title.TLabel").pack(pady=(0, 10))
        ttk.Label(main, text="3D-Roboter RL auf Windows & Ubuntu mit GPU-Unterstützung", style="Subtitle.TLabel").pack(pady=(0, 40))

        # Status-Box
        self.venv_exists = os.path.exists(VENV_PYTHON)
        status_frame = ttk.Frame(main)
        status_frame.pack(fill=tk.X, pady=(0, 30))

        if self.venv_exists:
            ttk.Label(
                status_frame,
                text="✅ Umgebung bereit",
                foreground="#2ecc71",
                font=("Segoe UI" if platform.system() == "Windows" else "Arial", 11, "bold"),
            ).pack()
        else:
            ttk.Label(
                status_frame,
                text="⚠️ Umgebung noch nicht eingerichtet — bitte Main.py zuerst ausführen",
                foreground="#e74c3c",
                font=("Segoe UI" if platform.system() == "Windows" else "Arial", 11, "bold"),
            ).pack()

        # === VERSIONEN ===
        ttk.Label(main, text="📋 Versionen wählen:", font=("Segoe UI" if platform.system() == "Windows" else "Arial", 13, "bold"), foreground="#3498db").pack(anchor=tk.W, pady=(0, 20))

        # V1
        self._add_version_card(
            main,
            "V1 — Einfacher Arm auf Rädern",
            "3-Achsen-Arm lernt Würfel greifen und in Box legen",
            "~22% Erfolgsrate, CPU-ready",
            self.start_v1_training,
            self.start_v1_demo,
            self.start_v1_eval,
        )

        ttk.Separator(main, orient="horizontal").pack(fill=tk.X, pady=30)

        # V2
        self._add_version_card(
            main,
            "V2 — Arm mit 3 Werkzeugen",
            "Greifer, Sauger, Schaufel — 3 Objekte, 3 Boxen, Hindernisse",
            "Werkzeugwechsel, Curriculum, Reward Shaping",
            self.start_v2_training,
            self.start_v2_demo,
            self.start_v2_eval,
        )

        ttk.Separator(main, orient="horizontal").pack(fill=tk.X, pady=30)

        # V3
        self._add_version_card(
            main,
            "V3 — Roboterhund (Quadruped)",
            "12 DOF, lernt vorwärts laufen ohne umzufallen",
            "Isaac Lab Ready, RTX GPU empfohlen",
            self.start_v3_training,
            self.start_v3_demo,
            self.start_v3_eval,
        )

        # === LOG-BEREICH ===
        ttk.Label(main, text="📝 Ausgabe:", font=("Segoe UI" if platform.system() == "Windows" else "Arial", 13, "bold"), foreground="#3498db").pack(anchor=tk.W, pady=(40, 10))
        self.log = scrolledtext.ScrolledText(
            main,
            height=12,
            bg="#34495e",
            fg="#ecf0f1",
            insertbackground="#ecf0f1",
            font=("Consolas" if platform.system() == "Windows" else "Courier", 9),
        )
        self.log.pack(fill=tk.BOTH, expand=True, pady=(0, 20))

        # Status
        self.status = ttk.Label(
            main,
            text="Bereit" if self.venv_exists else "Setup erforderlich",
            foreground="#2ecc71" if self.venv_exists else "#e74c3c",
            font=("Segoe UI" if platform.system() == "Windows" else "Arial", 10, "bold"),
        )
        self.status.pack(anchor=tk.W)

    def _add_version_card(self, parent, title, subtitle, details, cb_train, cb_demo, cb_eval):
        card = ttk.Frame(parent)
        card.pack(fill=tk.X, pady=15)

        # Titel und Beschreibung
        ttk.Label(card, text=title, font=("Segoe UI" if platform.system() == "Windows" else "Arial", 12, "bold"), foreground="#ecf0f1").pack(anchor=tk.W)
        ttk.Label(card, text=subtitle, foreground="#bdc3c7").pack(anchor=tk.W)
        ttk.Label(card, text=details, foreground="#95a5a6", font=("Segoe UI" if platform.system() == "Windows" else "Arial", 9)).pack(anchor=tk.W, pady=(0, 10))

        # Buttons
        btn_frame = ttk.Frame(card)
        btn_frame.pack(fill=tk.X, pady=(0, 10))

        state = "normal" if self.venv_exists else "disabled"

        ttk.Button(btn_frame, text="🚀 Training", command=cb_train, width=18, state=state).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_frame, text="🎮 Demo", command=cb_demo, width=14, state=state).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_frame, text="📊 Eval", command=cb_eval, width=14, state=state).pack(side=tk.LEFT)

    def _log(self, text: str):
        self.log.insert(tk.END, text + "\n")
        self.log.see(tk.END)
        self.update_idletasks()

    def _run_script(self, script_name: str, label: str):
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
                f"Skript nicht gefunden: {script_name}\nErwartet unter: {SCRIPT_DIR}",
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

            for line in proc.stdout:
                self._log(line.rstrip())

            proc.wait()
            if proc.returncode == 0:
                self.status.config(text=f"{label} abgeschlossen ✅", foreground="#2ecc71")
                self._log("-" * 50)
                self._log(f">>> {label} erfolgreich beendet.\n")
            else:
                self.status.config(text=f"{label} Fehler ❌", foreground="#e74c3c")
                self._log("-" * 50)
                self._log(f">>> {label} mit Fehlercode {proc.returncode} beendet.\n")

        except Exception as e:
            self.status.config(text="Fehler ❌", foreground="#e74c3c")
            self._log(f">>> Fehler: {e}\n")
            messagebox.showerror("Fehler", str(e))

    # === V1 ===
    def start_v1_training(self): self._run_script("train.py", "V1 Training")
    def start_v1_demo(self): self._run_script("demo.py", "V1 Demo")
    def start_v1_eval(self): self._run_script("eval.py", "V1 Eval")

    # === V2 ===
    def start_v2_training(self): self._run_script("train_v2.py", "V2 Training")
    def start_v2_demo(self): self._run_script("demo.py", "V2 Demo")
    def start_v2_eval(self): self._run_script("eval.py", "V2 Eval")

    # === V3 ===
    def start_v3_training(self): self._run_script("train_v3.py", "V3 Training")
    def start_v3_demo(self): self._run_script("demo.py", "V3 Demo")
    def start_v3_eval(self): self._run_script("eval.py", "V3 Eval")


if __name__ == "__main__":
    app = RLLauncher()
    app.mainloop()
