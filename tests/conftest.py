"""fixtures for mono3d tests"""

from pathlib import Path
from typing import Iterable

import pytest

from mono3d import CharucoBoard
from mono3d.charuco_board_detection import CharucoBoardDetection


@pytest.fixture
def charuco_board() -> CharucoBoard:
    """
    Fixture to create a CharucoBoard instance
    """
    return CharucoBoard()


@pytest.fixture
def detection_answer_dir() -> Path:
    """
    Fixture to answer directory for the detection results
    """

    detection_answer_dir = Path("tests/data/cam_calib_charuco_images/detection_answers")
    assert (
        detection_answer_dir.exists()
    ), f"detection_answer_dir {detection_answer_dir} does not exist"
    return detection_answer_dir


@pytest.fixture
def detections(detection_answer_dir) -> Iterable[CharucoBoardDetection]:
    """
    Fixture to load the charuco board image
    """
    detection_files = list(detection_answer_dir.glob("*.npz"))
    assert detection_files, "detection_files should not be empty"
    return [CharucoBoardDetection.load_from(file) for file in detection_files]


@pytest.fixture
def calibration_image_dir() -> Path:
    """
    Fixture to load the charuco board image
    """
    calibration_image_dir = Path("tests/data/cam_calib_charuco_images")
    assert (
        calibration_image_dir.exists()
    ), f"calibration_image_dir {calibration_image_dir} does not exist"
    return calibration_image_dir


@pytest.fixture
def calibration_image_paths(calibration_image_dir) -> Iterable[Path]:
    """
    Fixture to load the charuco board image
    """
    calibration_image_paths = list(calibration_image_dir.glob("*.jpg"))
    assert calibration_image_paths, "calibration_image_paths should not be empty"
    return calibration_image_paths
