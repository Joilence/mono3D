"""Unit tests for the CameraParameter class"""

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pytest

from mono3d import CharucoBoard, CameraParameter
from mono3d.charuco_board_detection import CharucoBoardDetection


@pytest.fixture
def calibration_answer_dir() -> Path:
    """
    Fixture to answer directory for the calibration results
    """

    calibration_answer_dir = Path("tests/images/cam_calib_charuco_images/calibration_answer")
    assert calibration_answer_dir.exists(), \
        f"calibration_answer_dir {calibration_answer_dir} does not exist"
    return calibration_answer_dir


@pytest.fixture
def calibration_answer(calibration_answer_dir) -> CameraParameter:
    """
    Fixture to load the calibration answer
    """
    return CameraParameter.load_from(calibration_answer_dir / "calibration_result.npz")


# add edge cases for calibrate_camera

def test_calibrate_camera_empty_detections(
        charuco_board: CharucoBoard,
        calibration_image_paths: Iterable[Path],
):
    # Arrange
    detections: Iterable[CharucoBoardDetection] = []

    # Act
    calib_res = charuco_board.calibrate_camera((100, 100), detections)

    # Assert
    assert calib_res is None, f"Failed on empty detections - calibration_result should be None, got:\n{calib_res}"


def test_calibrate_camera_happy_path(
        charuco_board: CharucoBoard,
        detections: Iterable[CharucoBoardDetection],
        calibration_image_paths: Iterable[Path],
        calibration_answer: CameraParameter,
):
    # Arrange

    # Take the first image to get the image size
    image_size = cv2.imread(str(next(iter(calibration_image_paths)))).shape[:2]  # TODO: could be a fixture

    calib_res_dir = Path("tests/images/cam_calib_charuco_images/calibration_result")
    calib_res_dir.mkdir(exist_ok=True)
    calib_ans = calibration_answer

    # Act
    calib_res = charuco_board.calibrate_camera(image_size, detections)

    # Assert
    assert calib_res is not None, "Failed on happy path - calibration_result should not be None"
    calib_res.save_to(calib_res_dir / "calibration_result.npz")
    assert np.allclose(calib_res.K, calib_ans.K), \
        "calibration_result.intrinsic_mat should be equal to calibration_answer.intrinsic_mat, " \
        f"got result:\n{calib_res.K}\nwhile answer is:\n{calib_ans.K}"
    assert np.allclose(calib_res.distortion_coeffs, calib_ans.distortion_coeffs), \
        "calibration_result.distortion_coeffs should be equal to calibration_answer.distortion_coeffs, " \
        f"got result:\n{calib_res.distortion_coeffs}\nwhile answer is:\n{calib_ans.distortion_coeffs}"
