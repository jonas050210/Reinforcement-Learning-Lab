#!/usr/bin/env python3
"""
================================================================================
  Reinforcement-Learning-Lab — GUI-Launcher (Windows 11 & Ubuntu)
================================================================================
Öffnet ein echtes GUI-Fenster (Tkinter) mit Buttons für alle 3 Versionen:

  V1 — Einfacher Arm auf Rädern
       3-Achsen-Arm lernt Würfel greifen und in Box legen
       → train.py, demo.py, eval.py

  V2 — Arm mit 3 Werkzeugen
       Greifer, Sauger, Schaufel — 3 Objekte, 3 Boxen, Hindernisse
       → train_v2.py, demo_v2.py, eval_v2.py

  V3 — Roboterhund (Quadruped)
       12 DOF, lernt vorwärts laufen ohne umzufallen
       → train_v3.py, demo_v3.py, eval_v3.py

Jeder Button startet das passende Python-Skript im Hintergrund.
Training = KI lernt die Aufgabe (dauert länger, speichert Modell)
Demo     = Trainiertes Modell live ansehen (MuJoCo-Fenster)
Eval     = Modell auswerten mit Statistiken (Erfolgsrate, Distanz)

Kein Terminal nötig — einfach python gui_launcher.py oder doppelklick.
================================================================================
"""

import subprocess
import sys
import os
import platform
import threading
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

        self.title("Reinforcement-Learning-Lab — Roboterarm & Hund (RL)")
        self.geometry("900x750")
        self.configure(bg="#1e2a38")

        # Stil
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TFrame", background="#1e2a38")
        style.configure(
            "TLabel",
            background="#1e2a38",
            foreground="#ecf0f1",
            font=("Segoe UI" if platform.system() == "Windows" else "Arial", 11),
        )
        style.configure(
            "Title.TLabel",
            font=("Segoe UI" if platform.system() == "Windows" else "Arial", 22, "bold"),
            foreground="#3498db",
        )
        style.configure(
            "Subtitle.TLabel",
            font=("Segoe UI" if platform.system() == "Windows" else "Arial", 12),
            foreground="#bdc3c7",
        )
        style.configure(
            "Card.TLabel",
            background="#2c3e50",
            foreground="#ecf0f1",
            font=("Segoe UI" if platform.system() == "Windows" else "Arial", 11),
        )
        style.configure(
            "TButton",
            font=("Segoe UI" if platform.system() == "Windows" else "Arial", 10, "bold"),
            padding=10,
        )

        self._build_ui()

    def _build_ui(self):
        # Hauptcontainer mit Scroll
        container = ttk.Frame(self)
        container.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(container, bg="#1e2a38", highlightthickness=0)
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

        main = ttk.Frame(scrollable_frame, padding="30")
        main.pack(fill=tk.BOTH, expand=True)

        # ================= TITEL =================
        title_frame = ttk.Frame(main)
        title_frame.pack(fill=tk.X, pady=(0, 30))

        ttk.Label(
            title_frame,
            text="🤖 Reinforcement-Learning-Lab",
            style="Title.TLabel",
        ).pack()
        ttk.Label(
            title_frame,
            text="Roboterarm & Roboterhund — KI lernt Greifen & Laufen",
            style="Subtitle.TLabel",
        ).pack(pady=(5, 0))

        # ================= HARDWARE INFO =================
        hw_frame = ttk.Frame(main)
        hw_frame.pack(fill=tk.X, pady=(0, 30))

        ttk.Label(
            hw_frame,
            text="🖥️  Ihr System:",
            font=("Segoe UI" if platform.system() == "Windows" else "Arial", 11, "bold"),
            foreground="#3498db",
        ).pack(anchor=tk.W)

        gpu_ok = self._check_gpu()
        hw_text = f"  CPU: {platform.processor() or 'Unbekannt'}"
        if gpu_ok:
            hw_text += "\n  GPU: NVIDIA RTX erkannt (CUDA verfügbar — Training wird beschleunigt)"
        else:
            hw_text += "\n  GPU: Keine NVIDIA-GPU erkannt — Training läuft auf CPU (funktioniert trotzdem)"
        ttk.Label(
            hw_frame,
            text=hw_text,
            style="Card.TLabel",
        ).pack(anchor=tk.W, pady=(5, 0))

        # ================= SETUP STATUS =================
        self.venv_exists = os.path.exists(VENV_PYTHON)
        status_frame = ttk.Frame(main)
        status_frame.pack(fill=tk.X, pady=(0, 30))

        if self.venv_exists:
            ttk.Label(
                status_frame,
                text="✅  Umgebung bereit — alle Buttons funktionieren",
                foreground="#2ecc71",
                font=("Segoe UI" if platform.system() == "Windows" else "Arial", 11, "bold"),
            ).pack()
        else:
            ttk.Label(
                status_frame,
                text="⚠️  Umgebung noch nicht eingerichtet",
                foreground="#e74c3c",
                font=("Segoe UI" if platform.system() == "Windows" else "Arial", 11, "bold"),
            ).pack()
            ttk.Label(
                status_frame,
                text="  → Bitte zuerst 'setup.py' ausführen (erstellt .venv und installiert alle Pakete)",
                foreground="#bdc3c7",
                font=("Segoe UI" if platform.system() == "Windows" else "Arial", 9),
            ).pack()

        # ================= LEGENDE =================
        legend_frame = ttk.Frame(main)
        legend_frame.pack(fill=tk.X, pady=(0, 20))

        ttk.Label(
            legend_frame,
            text="📋  Was bedeuten die Buttons?",
            font=("Segoe UI" if platform.system() == "Windows" else "Arial", 12, "bold"),
            foreground="#3498db",
        ).pack(anchor=tk.W, pady=(0, 8))

        legend_items = [
            ("🚀  Training", "KI lernt die Aufgabe (PPO-Algorithmus). Dauert länger, speichert das trainierte Modell ab."),
            ("🎮  Demo", "Trainiertes Modell live im MuJoCo-Fenster ansehen. Sieht man den Roboter in 3D arbeiten."),
            ("📊  Eval", "Modell auswerten mit Statistiken: Erfolgsrate, Distanz, Belohnung."),
        ]
        for title, desc in legend_items:
            item_frame = ttk.Frame(legend_frame)
            item_frame.pack(fill=tk.X, pady=2)
            ttk.Label(
                item_frame,
                text=title,
                font=("Segoe UI" if platform.system() == "Windows" else "Arial", 10, "bold"),
                foreground="#f39c12",
            ).pack(side=tk.LEFT)
            ttk.Label(
                item_frame,
                text="  —  " + desc,
                foreground="#bdc3c7",
                font=("Segoe UI" if platform.system() == "Windows" else "Arial", 9),
            ).pack(side=tk.LEFT)

        # ================= VERSIONEN =================
        ttk.Label(
            main,
            text="📦  Versionen wählen:",
            font=("Segoe UI" if platform.system() == "Windows" else "Arial", 13, "bold"),
            foreground="#3498db",
        ).pack(anchor=tk.W, pady=(30, 20))

        # V1
        self._add_version_card(
            main,
            "V1 — Einfacher Arm auf Rädern",
            "3-Achsen-Arm lernt Würfel greifen und in Box legen",
            "~22% Erfolgsrate  •  CPU-ready  •  train.py / demo.py / eval.py",
            self.start_v1_training,
            self.start_v1_demo,
            self.start_v1_eval,
        )

        ttk.Separator(main, orient="horizontal").pack(fill=tk.X, pady=25)

        # V2
        self._add_version_card(
            main,
            "V2 — Arm mit 3 Werkzeugen",
            "Greifer, Sauger, Schaufel — 3 Objekte, 3 Boxen, Hindernisse",
            "Werkzeugwechsel  •  Curriculum  •  Reward Shaping  •  train_v2.py / demo_v2.py / eval_v2.py",
            self.start_v2_training,
            self.start_v2_demo,
            self.start_v2_eval,
        )

        ttk.Separator(main, orient="horizontal").pack(fill=tk.X, pady=25)

        # V3
        self._add_version_card(
            main,
            "V3 — Roboterhund (Quadruped)",
            "12 DOF, lernt vorwärts laufen ohne umzufallen",
            "Isaac Lab Ready  •  RTX GPU empfohlen  •  train_v3.py / demo_v3.py / eval_v3.py",
            self.start_v3_training,
            self.start_v3_demo,
            self.start_v3_eval,
        )

        # ================= LOG-BEREICH =================
        log_frame = ttk.Frame(main)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(30, 0))

        ttk.Label(
            log_frame,
            text="📝  Ausgabe:",
            font=("Segoe UI" if platform.system() == "Windows" else "Arial", 12, "bold"),
            foreground="#3498db",
        ).pack(anchor=tk.W)
        self.log = scrolledtext.ScrolledText(
            log_frame,
            height=10,
            bg="#2c3e50",
            fg="#ecf0f1",
            insertbackground="#ecf0f1",
            font=("Consolas" if platform.system() == "Windows" else "Courier", 9),
        )
        self.log.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

        # Status
        self.status = ttk.Label(
            main,
            text="Bereit" if self.venv_exists else "Setup erforderlich — bitte setup.py ausführen",
            foreground="#2ecc71" if self.venv_exists else "#e74c3c",
            font=("Segoe UI" if platform.system() == "Windows" else "Arial", 10, "bold"),
        )
        self.status.pack(anchor=tk.W, pady=(10, 0))

    def _add_version_card(self, parent, title, subtitle, details, cb_train, cb_demo, cb_eval):
        card = ttk.Frame(parent)
        card.pack(fill=tk.X, pady=10)

        # Karten-Hintergrund-Effekt durch Rahmen
        card_inner = ttk.Frame(card)
        card_inner.pack(fill=tk.X, padx=2, pady=2)

        # Titel und Beschreibung
        ttk.Label(
            card_inner,
            text=title,
            font=("Segoe UI" if platform.system() == "Windows" else "Arial", 13, "bold"),
            foreground="#ecf0f1",
        ).pack(anchor=tk.W, pady=(8, 2))
        ttk.Label(
            card_inner,
            text=subtitle,
            foreground="#bdc3c7",
            font=("Segoe UI" if platform.system() == "Windows" else "Arial", 10),
        ).pack(anchor=tk.W, pady=(0, 2))
        ttk.Label(
            card_inner,
            text=details,
            foreground="#95a5a6",
            font=("Segoe UI" if platform.system() == "Windows" else "Arial", 8),
        ).pack(anchor=tk.W, pady=(0, 8))

        # Buttons
        btn_frame = ttk.Frame(card_inner)
        btn_frame.pack(fill=tk.X, pady=(0, 8))

        ttk.Button(
            btn_frame,
            text="🚀  Training",
            command=cb_train,
            width=16,
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(
            btn_frame,
            text="🎮  Demo",
            command=cb_demo,
            width=12,
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(
            btn_frame,
            text="📊  Eval",
            command=cb_eval,
            width=12,
        ).pack(side=tk.LEFT)

        if not self.venv_exists:
            ttk.Label(
                btn_frame,
                text="  ⚠ Setup zuerst ausführen",
                foreground="#e74c3c",
                font=("Segoe UI" if platform.system() == "Windows" else "Arial", 8),
            ).pack(side=tk.LEFT, padx=(10, 0))

    def _check_gpu(self) -> bool:
        try:
            result = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    def _log(self, text: str):
        self.log.insert(tk.END, text + "\n")
        self.log.see(tk.END)
        self.update_idletasks()

    def _run_script(self, script_name: str, label: str):
        if not self.venv_exists:
            messagebox.showerror(
                "Umgebung nicht bereit",
                "Virtuelle Umgebung (.venv) nicht gefunden!\n\n"
                "Bitte zuerst setup.py ausführen:\n"
                "  python setup.py\n"
                "Das installiert alle Pakete und startet dann automatisch die GUI.\n\n"
                "Erst danach funktionieren die Buttons hier.",
            )
            return

        script_path = os.path.join(SCRIPT_DIR, script_name)
        if not os.path.exists(script_path):
            messagebox.showerror(
                "Skript nicht gefunden",
                f"Skript nicht gefunden: {script_name}\n"
                f"Erwartet unter: {SCRIPT_DIR}",
            )
            return

        # Buttons deaktivieren während Laufzeit (GUI friert nicht ein dank Thread)
        self._set_buttons_enabled(False)

        self.status.config(text=f"{label} läuft...", foreground="#f39c12")
        self._log(f">>> Starte {label}...")
        self._log(f">>> Python: {VENV_PYTHON}")
        self._log(f">>> Skript: {script_path}")
        self._log("-" * 50)

        def _target():
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
                    self.after(0, lambda l=line: self._log(l.rstrip()))
                proc.wait()
                rc = proc.returncode
                self.after(0, lambda: self._on_finished(label, rc))
            except Exception as e:
                self.after(0, lambda: self._on_error(label, e))

        thread = threading.Thread(target=_target, daemon=True)
        thread.start()

    def _on_finished(self, label: str, rc: int):
        self._set_buttons_enabled(True)
        self._log("-" * 50)
        if rc == 0:
            self.status.config(text=f"{label} abgeschlossen ✅", foreground="#2ecc71")
            self._log(f">>> {label} erfolgreich beendet.\n")
        else:
            self.status.config(text=f"{label} Fehler ❌", foreground="#e74c3c")
            self._log(f">>> {label} mit Fehlercode {rc} beendet.\n")

    def _on_error(self, label: str, e: Exception):
        self._set_buttons_enabled(True)
        self.status.config(text="Fehler ❌", foreground="#e74c3c")
        self._log(f">>> Fehler: {e}\n")
        messagebox.showerror("Fehler", str(e))

    def _set_buttons_enabled(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        for widget in self.winfo_children():
            self._set_ttk_button_state(widget, state)

    def _set_ttk_button_state(self, widget, state: str):
        if isinstance(widget, ttk.Button):
            try:
                widget.configure(state=state)
            except Exception:
                pass
        for child in widget.winfo_children():
            self._set_ttk_button_state(child, state)

    # === V1 ===
    def start_v1_training(self): self._run_script("train.py", "V1 Training")
    def start_v1_demo(self): self._run_script("demo.py", "V1 Demo")
    def start_v1_eval(self): self._run_script("eval.py", "V1 Eval")

    # === V2 ===
    def start_v2_training(self): self._run_script("train_v2.py", "V2 Training")
    def start_v2_demo(self): self._run_script("demo_v2.py", "V2 Demo")
    def start_v2_eval(self): self._run_script("eval_v2.py", "V2 Eval")

    # === V3 ===
    def start_v3_training(self): self._run_script("train_v3.py", "V3 Training")
    def start_v3_demo(self): self._run_script("demo_v3.py", "V3 Demo")
    def start_v3_eval(self): self._run_script("eval_v3.py", "V3 Eval")


if __name__ == "__main__":
    app = RLLauncher()
    app.mainloop()