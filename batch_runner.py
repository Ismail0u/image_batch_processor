"""
batch_runner.py
---------------
Orchestrateur du traitement par lots.
Expose deux modes : séquentiel et parallèle (multiprocessing.Pool).
"""

import os
import time
from multiprocessing import Pool, Value, cpu_count
from pathlib import Path
from typing import List, Tuple, Optional

from image_processor import FilterType, ProcessingResult, TaskPayload, process_image


# ─────────────────────────────────────────────
#  Utilitaires
# ─────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def collect_images(input_dir: str) -> List[str]:
    """Retourne tous les chemins d'images supportées dans input_dir."""
    paths = []
    for entry in sorted(Path(input_dir).iterdir()):
        if entry.is_file() and entry.suffix.lower() in SUPPORTED_EXTENSIONS:
            paths.append(str(entry))
    return paths


def build_tasks(
    image_paths: List[str],
    output_dir: str,
    filter_type: FilterType,
    filter_params: dict,
    suffix: str = "",
) -> List[TaskPayload]:
    """Construit la liste des TaskPayload à partir des chemins d'images."""
    tasks = []
    for path in image_paths:
        filename  = Path(path).stem + suffix + Path(path).suffix
        out_path  = os.path.join(output_dir, filename)
        tasks.append(TaskPayload(
            input_path=path,
            output_path=out_path,
            filter_type=filter_type,
            filter_params=filter_params,
        ))
    return tasks


# ─────────────────────────────────────────────
#  Modes d'exécution
# ─────────────────────────────────────────────

def run_sequential(
    tasks: List[TaskPayload],
    *,
    progress_value: Optional[Value] = None
) -> Tuple[List[ProcessingResult], float]:
    """
    Traitement séquentiel — baseline de comparaison.
    Incrémente progress_value (si fourni) après chaque image traitée.
    """
    t0 = time.perf_counter()
    results = []
    for task in tasks:
        res = process_image(task)
        results.append(res)
        if progress_value is not None:
            with progress_value.get_lock():
                progress_value.value += 1
    elapsed = time.perf_counter() - t0
    return results, elapsed


def run_parallel(
    tasks: List[TaskPayload],
    num_workers: int,
    chunk_size: int = None,
    *,
    progress_value: Optional[Value] = None,
) -> Tuple[List[ProcessingResult], float]:
    """
    Traitement parallèle avec multiprocessing.Pool.

    progress_value : si fourni, sera incrémenté atomiquement
    à chaque image terminée. (keyword-only pour éviter toute confusion avec chunk_size).
    """
    effective_workers = min(num_workers, cpu_count())

    if chunk_size is None:
        chunk_size = max(1, len(tasks) // (effective_workers * 4))

    t0 = time.perf_counter()
    with Pool(processes=effective_workers) as pool:
        # imap_unordered permet d'incrémenter la progression au fil de l'eau
        results = []
        for res in pool.imap_unordered(process_image, tasks, chunksize=chunk_size):
            results.append(res)
            if progress_value is not None:
                with progress_value.get_lock():
                    progress_value.value += 1
    elapsed = time.perf_counter() - t0

    return results, elapsed


# ─────────────────────────────────────────────
#  Rapport de résultats
# ─────────────────────────────────────────────

def summarize(
    results: List[ProcessingResult],
    elapsed: float,
    mode_label: str,
) -> dict:
    """Agrège les résultats en métriques de performance."""
    successes = [r for r in results if r.success]
    failures  = [r for r in results if not r.success]

    durations = [r.duration_ms for r in successes]
    avg_ms    = sum(durations) / len(durations) if durations else 0
    total_kb  = sum(r.file_size_kb for r in successes if r.file_size_kb)

    return {
        "mode":           mode_label,
        "total_images":   len(results),
        "successes":      len(successes),
        "failures":       len(failures),
        "wall_time_s":    round(elapsed, 3),
        "avg_per_img_ms": round(avg_ms, 2),
        "throughput_img_s": round(len(successes) / elapsed, 2) if elapsed > 0 else 0,
        "total_output_kb": round(total_kb, 1),
        "failed_files":   [r.input_path for r in failures],
    }