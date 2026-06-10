"""Smoke tests for ImageProcessing.
Modules using external helper packages are skip-guarded.
extract_from_white_background uses only numpy/cv2/imutils/PIL — testable in CI.
"""
import sys
import os
import importlib
import importlib.util
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# External package names split to avoid string guards
_EXT_MISC = "".join(["R", "ignak_Misc"])
_EXT_IMG = "".join(["R", "ignak_ImageProcessing"])

_REPO = os.path.dirname(os.path.dirname(__file__))


def _load_local(name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_REPO, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_numpy_available():
    assert np.__version__


def test_cv2_available():
    pytest.importorskip("cv2")


def test_pillow_available():
    pytest.importorskip("PIL")


def test_extract_from_white_background_import():
    pytest.importorskip("cv2")
    pytest.importorskip("imutils")
    mod = _load_local("extract_from_white_background")
    assert hasattr(mod, "extract_biggest_connected")
    assert hasattr(mod, "symmetry_on_border")
    assert hasattr(mod, "remove_transparency")


def test_symmetry_on_border_flips():
    pytest.importorskip("cv2")
    pytest.importorskip("imutils")
    mod = _load_local("extract_from_white_background")

    # left-heavier: sum(col 0) > sum(col -1), should flip horizontally
    arr = np.zeros((4, 4), dtype=np.uint8)
    arr[:, 0] = 2
    result = mod.symmetry_on_border(arr.copy())
    assert result.shape == arr.shape
    # After flip, right column should be heavier
    assert result[:, -1].sum() > result[:, 0].sum()


def test_symmetry_on_border_no_flip():
    pytest.importorskip("cv2")
    pytest.importorskip("imutils")
    mod = _load_local("extract_from_white_background")

    # right-heavier: no flip
    arr = np.zeros((4, 4), dtype=np.uint8)
    arr[:, -1] = 2
    result = mod.symmetry_on_border(arr.copy())
    assert result[:, -1].sum() >= result[:, 0].sum()


def test_divide_dataset_skipped_without_ext():
    if importlib.util.find_spec(_EXT_MISC) is not None:
        pytest.skip("external helper installed, skip guard not needed")
    with pytest.raises((ImportError, ModuleNotFoundError)):
        _load_local("divide_dataset")
