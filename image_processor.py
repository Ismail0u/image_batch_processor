"""
image_processor.py
------------------
Fonctions de traitement d'image (stateless, picklables pour multiprocessing).
Chaque fonction prend un TaskPayload et retourne un ProcessingResult.
"""

import os
import time
import traceback
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import numpy as np


# ─────────────────────────────────────────────
#  Types
# ─────────────────────────────────────────────

class FilterType(str, Enum):
    GRAYSCALE       = "grayscale"
    BLUR            = "blur"
    EDGE_DETECTION  = "edge"
    SHARPEN         = "sharpen"
    EMBOSS          = "emboss"


@dataclass
class TaskPayload:
    """Unité de travail envoyée à chaque worker."""
    input_path:  str
    output_path: str
    filter_type: FilterType
    filter_params: dict   # paramètres supplémentaires (ex: kernel_size)


@dataclass
class ProcessingResult:
    """Résultat retourné par un worker."""
    input_path:   str
    output_path:  str
    success:      bool
    duration_ms:  float
    error:        Optional[str] = None
    file_size_kb: Optional[float] = None


# ─────────────────────────────────────────────
#  Kernels & filtres
# ─────────────────────────────────────────────

_SHARPEN_KERNEL = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
], dtype=np.float32)

_EMBOSS_KERNEL = np.array([
    [-2, -1,  0],
    [-1,  1,  1],
    [ 0,  1,  2]
], dtype=np.float32)


def _apply_grayscale(img: np.ndarray, params: dict) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # garde 3 canaux pour homogénéité


def _apply_blur(img: np.ndarray, params: dict) -> np.ndarray:
    k = params.get("kernel_size", 15)
    k = k if k % 2 == 1 else k + 1   # kernel impair obligatoire
    sigma = params.get("sigma", 0)
    return cv2.GaussianBlur(img, (k, k), sigma)


def _apply_edge_detection(img: np.ndarray, params: dict) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t1   = params.get("threshold1", 100)
    t2   = params.get("threshold2", 200)
    edges = cv2.Canny(gray, t1, t2)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def _apply_sharpen(img: np.ndarray, params: dict) -> np.ndarray:
    return cv2.filter2D(img, -1, _SHARPEN_KERNEL)


def _apply_emboss(img: np.ndarray, params: dict) -> np.ndarray:
    result = cv2.filter2D(img, -1, _EMBOSS_KERNEL)
    return cv2.add(result, np.ones_like(result) * 128)   # décalage neutre


_FILTER_MAP = {
    FilterType.GRAYSCALE:      _apply_grayscale,
    FilterType.BLUR:           _apply_blur,
    FilterType.EDGE_DETECTION: _apply_edge_detection,
    FilterType.SHARPEN:        _apply_sharpen,
    FilterType.EMBOSS:         _apply_emboss,
}


# ─────────────────────────────────────────────
#  Point d'entrée du worker  (doit être picklable → top-level)
# ─────────────────────────────────────────────

def process_image(task: TaskPayload) -> ProcessingResult:
    """
    Fonction appelée par chaque worker du Pool.
    Design: pas d'état global → thread/process-safe nativement.
    """
    t0 = time.perf_counter()

    try:
        # 1. Lecture
        img = cv2.imread(task.input_path)
        if img is None:
            raise ValueError(f"Impossible de lire l'image: {task.input_path}")

        # 2. Application du filtre
        filter_fn = _FILTER_MAP[task.filter_type]
        result    = filter_fn(img, task.filter_params)

        # 3. Sauvegarde
        os.makedirs(os.path.dirname(task.output_path), exist_ok=True)
        success = cv2.imwrite(task.output_path, result)
        if not success:
            raise IOError(f"Échec écriture: {task.output_path}")

        elapsed_ms  = (time.perf_counter() - t0) * 1000
        file_size   = os.path.getsize(task.output_path) / 1024

        return ProcessingResult(
            input_path=task.input_path,
            output_path=task.output_path,
            success=True,
            duration_ms=elapsed_ms,
            file_size_kb=file_size,
        )

    except Exception as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return ProcessingResult(
            input_path=task.input_path,
            output_path=task.output_path,
            success=False,
            duration_ms=elapsed_ms,
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        )