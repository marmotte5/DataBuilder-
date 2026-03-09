"""Fonctions utilitaires."""

import os
from pathlib import Path

from dataset_sorter.constants import SAFE_NAME_RE


def sanitize_folder_name(name: str) -> str:
    """Nettoie un nom de dossier en supprimant les caractères non sûrs."""
    cleaned = SAFE_NAME_RE.sub("", name).strip()
    return cleaned if cleaned else "bucket"


def is_path_inside(child: Path, parent: Path) -> bool:
    """Vérifie qu'un chemin résolu est à l'intérieur d'un autre (anti path-traversal)."""
    try:
        child_resolved = child.resolve()
        parent_resolved = parent.resolve()
        return (
            str(child_resolved).startswith(str(parent_resolved) + os.sep)
            or child_resolved == parent_resolved
        )
    except (OSError, ValueError):
        return False


def validate_paths(source: str, output: str) -> tuple[bool, str]:
    """Valide la paire source/output. Retourne (ok, message_erreur)."""
    if not source or not Path(source).is_dir():
        return False, "Le dossier source n'existe pas ou n'est pas défini."
    if not output:
        return False, "Le dossier de sortie n'est pas défini."

    src = Path(source).resolve()
    out = Path(output).resolve()

    if src == out:
        return False, "Le dossier source et le dossier de sortie sont identiques."
    if is_path_inside(out, src):
        return False, (
            "Le dossier de sortie est à l'intérieur du dossier source. "
            "Cela pourrait corrompre vos données."
        )
    if is_path_inside(src, out):
        return False, (
            "Le dossier source est à l'intérieur du dossier de sortie. "
            "Cela pourrait corrompre vos données."
        )
    return True, ""
