"""
main.py
-------
Interface graphique Tkinter professionnelle pour le traitement d'images par lots.
Ajouts :
- Bouton "Générer images test" (crée des images synthétiques)
- Sélection du filtre par boutons exclusifs (au lieu d'une Combobox)
- Dashboard temps réel pendant le traitement
- Affichage final des métriques dans un panneau dédié
"""

import json
import os
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from multiprocessing import Value
import cv2
import numpy as np

from batch_runner import (
    build_tasks,
    collect_images,
    run_parallel,
    run_sequential,
    summarize,
)
from image_processor import FilterType


# ─────────────────────────────────────────────
#  Fonctions utilitaires
# ─────────────────────────────────────────────

def generate_default_images(output_dir: str = "./generated_images", num: int = 10, size: int = 512):
    """Crée des images RGB aléatoires pour tester l'application."""
    os.makedirs(output_dir, exist_ok=True)
    for i in range(num):
        img = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
        # Dessiner des formes pour avoir du contenu détectable par les filtres
        cv2.circle(img, (size//2, size//2), size//3, (0, 0, 255), -1)
        cv2.putText(img, str(i+1), (size//4, size//4), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
        cv2.imwrite(os.path.join(output_dir, f"test_{i+1:03d}.png"), img)
    return output_dir


def format_time(seconds: float) -> str:
    """Formate un temps en secondes en chaîne lisible."""
    if seconds < 1:
        return f"{seconds*1000:.0f} ms"
    elif seconds < 60:
        return f"{seconds:.1f} s"
    else:
        mins, sec = divmod(seconds, 60)
        return f"{int(mins)}m {sec:.0f}s"


# ─────────────────────────────────────────────
#  Classe GUI
# ─────────────────────────────────────────────

class ImageBatchProcessorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Traitement d'Images par Lots – Dashboard")
        self.geometry("750x650")
        self.minsize(700, 600)  # fenêtre responsive minimum

        # Variables de contrôle
        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar(value="./output")
        self.active_filter = tk.StringVar(value=FilterType.GRAYSCALE.value)  # filtre sélectionné
        self.workers_var = tk.IntVar(value=4)
        self.mode_var = tk.StringVar(value="parallel")
        self.kernel_var = tk.IntVar(value=15)

        # Variables pour le dashboard
        self.total_images = 0
        self.processed_images = 0
        self.start_time = 0.0
        self.timer_running = False

        # Widgets
        self.create_widgets()

    # ------------------------------------------------------------
    #  Construction de l'interface
    # ------------------------------------------------------------
    def create_widgets(self):
        # --- Frame principal (marges) ---
        main_frame = ttk.Frame(self, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # --- Dossier source ---
        src_frame = ttk.LabelFrame(main_frame, text="1. Dossier source", padding="5")
        src_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0,10))
        src_frame.columnconfigure(1, weight=1)

        ttk.Entry(src_frame, textvariable=self.input_dir).grid(row=0, column=0, sticky="ew", padx=(0,5))
        ttk.Button(src_frame, text="Parcourir", command=self.browse_input).grid(row=0, column=1, padx=2)
        ttk.Button(src_frame, text="Générer images test", command=self.generate_and_set_input).grid(row=0, column=2, padx=2)

        # --- Dossier de sortie ---
        out_frame = ttk.LabelFrame(main_frame, text="2. Dossier de sortie", padding="5")
        out_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0,10))
        out_frame.columnconfigure(0, weight=1)

        ttk.Entry(out_frame, textvariable=self.output_dir).grid(row=0, column=0, sticky="ew", padx=(0,5))
        ttk.Button(out_frame, text="Parcourir", command=self.browse_output).grid(row=0, column=1, padx=2)

        # --- Choix du filtre (boutons exclusifs) ---
        filter_frame = ttk.LabelFrame(main_frame, text="3. Filtre à appliquer (un seul)", padding="5")
        filter_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0,10))
        filter_frame.columnconfigure(0, weight=1)
        filter_frame.columnconfigure(1, weight=1)
        filter_frame.columnconfigure(2, weight=1)

        self.filter_buttons = {}
        for idx, filter_type in enumerate(FilterType):
            btn = ttk.Button(
                filter_frame,
                text=filter_type.value.capitalize(),
                command=lambda ft=filter_type: self.select_filter(ft.value)
            )
            btn.grid(row=0, column=idx, padx=5, pady=5, sticky="ew")
            self.filter_buttons[filter_type.value] = btn

        # Premier filtre sélectionné par défaut
        self.select_filter(FilterType.GRAYSCALE.value)

        # --- Options de traitement ---
        options_frame = ttk.LabelFrame(main_frame, text="4. Paramètres", padding="5")
        options_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0,10))
        options_frame.columnconfigure(0, weight=1)

        inner = ttk.Frame(options_frame)
        inner.grid(row=0, column=0, sticky="ew")
        inner.columnconfigure(0, weight=1)  # stretch colonne workers
        inner.columnconfigure(2, weight=1)  # stretch colonne kernel

        # Workers
        ttk.Label(inner, text="Workers (processus):").grid(row=0, column=0, sticky="w", padx=5)
        ttk.Spinbox(inner, from_=1, to=16, textvariable=self.workers_var, width=5).grid(row=0, column=1, padx=5)

        # Taille du kernel
        ttk.Label(inner, text="Taille kernel (impair):").grid(row=0, column=2, sticky="w", padx=(20,5))
        ttk.Spinbox(inner, from_=1, to=31, increment=2, textvariable=self.kernel_var, width=5).grid(row=0, column=3, padx=5)

        # Mode de traitement
        mode_frame = ttk.Frame(options_frame)
        mode_frame.grid(row=1, column=0, sticky="ew", pady=(10,0))
        ttk.Label(mode_frame, text="Mode:").pack(side="left")
        ttk.Radiobutton(mode_frame, text="Parallèle", variable=self.mode_var, value="parallel").pack(side="left", padx=5)
        ttk.Radiobutton(mode_frame, text="Séquentiel", variable=self.mode_var, value="sequential").pack(side="left", padx=5)
        ttk.Radiobutton(mode_frame, text="Benchmark", variable=self.mode_var, value="benchmark").pack(side="left", padx=5)

        # --- Bouton TRAITER ---
        self.traiter_btn = ttk.Button(main_frame, text="LANCER LE TRAITEMENT", command=self.start_processing)
        self.traiter_btn.grid(row=4, column=0, columnspan=2, pady=10, sticky="ew")

        # --- Barre de progression ---
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(0,10))

        # --- Dashboard temps réel ---
        dashboard_frame = ttk.LabelFrame(main_frame, text="5. Dashboard", padding="5")
        dashboard_frame.grid(row=6, column=0, columnspan=2, sticky="nsew", pady=(0,10))
        main_frame.rowconfigure(6, weight=1)  # permet d'étendre le dashboard

        # Grille interne pour les métriques
        self.dash_labels = {}
        metrics = [
            ("Images totales :", "total", "0"),
            ("Images traitées :", "processed", "0"),
            ("Temps écoulé :", "elapsed", "0.0 s"),
            ("Temps restant estimé :", "remaining", "--"),
            ("Débit (img/s) :", "throughput", "--"),
        ]
        for i, (label, key, default) in enumerate(metrics):
            ttk.Label(dashboard_frame, text=label).grid(row=i, column=0, sticky="w", padx=5, pady=2)
            lbl = ttk.Label(dashboard_frame, text=default, width=25, anchor="w")
            lbl.grid(row=i, column=1, sticky="w", padx=5, pady=2)
            self.dash_labels[key] = lbl

        # --- Zone de rapport final (texte défilant) ---
        self.results_text = tk.Text(main_frame, height=8, wrap="word", state="disabled")
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        self.results_text.grid(row=7, column=0, columnspan=2, sticky="nsew")
        scrollbar.grid(row=7, column=2, sticky="ns")
        main_frame.rowconfigure(7, weight=2)

    # ------------------------------------------------------------
    #  Méthodes de navigation et génération d'images
    # ------------------------------------------------------------
    def browse_input(self):
        dir_path = filedialog.askdirectory(title="Sélectionner le dossier contenant les images")
        if dir_path:
            self.input_dir.set(dir_path)

    def browse_output(self):
        dir_path = filedialog.askdirectory(title="Sélectionner le dossier de sortie")
        if dir_path:
            self.output_dir.set(dir_path)

    def generate_and_set_input(self):
        """Génère des images test dans un dossier choisi et le définit comme source."""
        save_dir = filedialog.askdirectory(title="Dossier où générer les images test")
        if not save_dir:
            return

        # Petite boîte de dialogue pour nombre/taille (simplifiée)
        settings = tk.Toplevel(self)
        settings.title("Paramètres de génération")
        settings.geometry("250x150")
        settings.transient(self)
        settings.grab_set()

        ttk.Label(settings, text="Nombre d'images :").pack(pady=5)
        nb_var = tk.IntVar(value=10)
        ttk.Spinbox(settings, from_=1, to=100, textvariable=nb_var, width=5).pack()

        ttk.Label(settings, text="Taille (pixels) :").pack(pady=5)
        size_var = tk.IntVar(value=512)
        ttk.Spinbox(settings, from_=64, to=2048, increment=64, textvariable=size_var, width=5).pack()

        def on_ok():
            try:
                num = nb_var.get()
                sz = size_var.get()
            except:
                messagebox.showerror("Erreur", "Valeurs invalides")
                return
            settings.destroy()
            # Génération
            final_dir = generate_default_images(save_dir, num, sz)
            self.input_dir.set(final_dir)
            messagebox.showinfo("Succès", f"{num} images {sz}x{sz} créées dans\n{final_dir}")

        ttk.Button(settings, text="Générer", command=on_ok).pack(pady=10)

    # ------------------------------------------------------------
    #  Sélection du filtre par boutons
    # ------------------------------------------------------------
    def select_filter(self, filter_value: str):
        """Active visuellement le bouton du filtre choisi et met à jour la variable."""
        self.active_filter.set(filter_value)
        # Style normal pour tous les boutons
        style = ttk.Style()
        style.configure("ActiveFilter.TButton", background="green")  # pas supporté partout, on change le relief
        # Méthode cross-platform : changer le relief
        for filt, btn in self.filter_buttons.items():
            if filt == filter_value:
                btn.state(['pressed'])  # effet enfoncé (visuel)
            else:
                btn.state(['!pressed'])

    # ------------------------------------------------------------
    #  Traitement asynchrone
    # ------------------------------------------------------------
    def start_processing(self):
        input_dir = self.input_dir.get()
        if not os.path.isdir(input_dir):
            messagebox.showerror("Erreur", f"Dossier introuvable : {input_dir}")
            return

        images = collect_images(input_dir)
        if not images:
            messagebox.showerror("Erreur", "Aucune image trouvée (.jpg, .png) dans ce dossier.")
            return

        # Verrouiller l'interface
        self.traiter_btn.config(state="disabled")
        self.clear_dashboard()
        self.clear_report()
        self.progress_var.set(0)

        mode = self.mode_var.get()
        if mode == "benchmark":
            self.progress_bar.config(maximum=len(images) * 2)
        else:
            self.progress_bar.config(maximum=len(images))

        # Démarrage du traitement dans un thread séparé
        threading.Thread(
            target=self.process_images,
            args=(images,),
            daemon=True
        ).start()

        # Démarrer la mise à jour du dashboard (temps réel)
        self.start_time = time.time()
        self.total_images = len(images)
        self.processed_images = 0
        self.timer_running = True
        self.update_dashboard_loop()

    def process_images(self, images):
        filter_type = FilterType(self.active_filter.get())
        filter_params = {"kernel_size": self.kernel_var.get()}
        mode = self.mode_var.get()
        workers = self.workers_var.get()
        output = self.output_dir.get()

        os.makedirs(output, exist_ok=True)
        progress_value = Value('i', 0)

        results_seq = results_par = None
        stats_seq = stats_par = None

        # Sequential
        if mode in ("sequential", "benchmark"):
            out_seq = os.path.join(output, "sequential")
            tasks = build_tasks(images, out_seq, filter_type, filter_params, suffix="_seq")
            results_seq, elapsed = run_sequential(tasks, progress_value=progress_value)
            stats_seq = summarize(results_seq, elapsed, "SÉQUENTIEL")

        # Parallel
        if mode in ("parallel", "benchmark"):
            out_par = os.path.join(output, "parallel")
            tasks = build_tasks(images, out_par, filter_type, filter_params, suffix="_par")
            results_par, elapsed = run_parallel(tasks, workers, progress_value=progress_value)
            stats_par = summarize(results_par, elapsed, f"PARALLÈLE ({workers} workers)")

        # Rapport final dans l'interface
        self.after(0, self.display_final_report, stats_seq, stats_par, mode, workers)

        # Mise à jour finale du dashboard
        self.timer_running = False
        self.after(0, self.final_dashboard_update)

    def update_dashboard_loop(self):
        """Mise à jour périodique des métriques du dashboard."""
        if not self.timer_running:
            return

        elapsed = time.time() - self.start_time
        self.dash_labels["elapsed"].config(text=format_time(elapsed))

        # Lecture de la progression
        current = self.progress_var.get()
        self.processed_images = int(current)
        self.dash_labels["processed"].config(text=f"{self.processed_images} / {self.total_images}")

        # Temps restant estimé
        if self.processed_images > 0 and self.total_images > 0:
            remaining = (elapsed / self.processed_images) * (self.total_images - self.processed_images)
            self.dash_labels["remaining"].config(text=format_time(remaining))
            throughput = self.processed_images / elapsed if elapsed > 0 else 0
            self.dash_labels["throughput"].config(text=f"{throughput:.2f} img/s")
        else:
            self.dash_labels["remaining"].config(text="--")
            self.dash_labels["throughput"].config(text="--")

        # Prochaine mise à jour dans 200 ms
        self.after(200, self.update_dashboard_loop)

    def final_dashboard_update(self):
        """Désactive le minuteur et affiche les valeurs finales."""
        self.timer_running = False
        elapsed = time.time() - self.start_time
        self.dash_labels["elapsed"].config(text=format_time(elapsed))
        self.dash_labels["remaining"].config(text="Terminé")
        self.traiter_btn.config(state="normal")

    def display_final_report(self, stats_seq, stats_par, mode, workers):
        """Affiche le rapport textuel final dans la zone de texte."""
        self.results_text.config(state="normal")
        self.results_text.delete(1.0, tk.END)

        sep = "─" * 52
        if stats_seq:
            report = f"\n{sep}\n"
            report += f"  Mode       : Séquentiel\n"
            report += f"  Images     : {stats_seq['successes']}/{stats_seq['total_images']}\n"
            report += f"  Temps      : {stats_seq['wall_time_s']:.2f}s\n"
            report += f"  Débit      : {stats_seq['throughput_img_s']:.2f} img/s\n"
            report += f"{sep}\n"
            self.results_text.insert(tk.END, report)

        if stats_par:
            report = f"\n{sep}\n"
            report += f"  Mode       : Parallèle ({workers} workers)\n"
            report += f"  Images     : {stats_par['successes']}/{stats_par['total_images']}\n"
            report += f"  Temps      : {stats_par['wall_time_s']:.2f}s\n"
            report += f"  Débit      : {stats_par['throughput_img_s']:.2f} img/s\n"
            report += f"{sep}\n"
            self.results_text.insert(tk.END, report)

        # Speedup si benchmark
        if mode == "benchmark" and stats_seq and stats_par and stats_seq["wall_time_s"] > 0:
            speedup = stats_seq["wall_time_s"] / stats_par["wall_time_s"]
            efficacy = (speedup / workers) * 100
            speedup_text = (
                f"\n{'═' * 52}\n"
                f"  SPEEDUP     : {speedup:.2f}x (max théorique {workers}x)\n"
                f"  EFFICACITÉ  : {efficacy:.1f}%\n"
                f"  (Overhead IPC inclus)\n"
                f"{'═' * 52}\n"
            )
            self.results_text.insert(tk.END, speedup_text)

        dossier = self.output_dir.get()
        self.results_text.insert(tk.END, f"\n✓ Résultats sauvegardés dans {dossier}\n")
        self.results_text.config(state="disabled")

    def clear_dashboard(self):
        self.dash_labels["total"].config(text=str(self.total_images))
        for key in ["processed", "elapsed", "remaining", "throughput"]:
            self.dash_labels[key].config(text="--" if key != "processed" else "0")

    def clear_report(self):
        self.results_text.config(state="normal")
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state="disabled")


# ─────────────────────────────────────────────
#  Point d'entrée
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = ImageBatchProcessorApp()
    app.mainloop()