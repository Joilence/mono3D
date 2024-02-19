"""Unit tests for the CameraParameter class"""

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pytest

from mono3d import CameraParameter, CharucoBoard
from mono3d.charuco_board_detection import CharucoBoardDetection


@pytest.fixture
def calibration_answer_dir() -> Path:
    """
    Fixture to answer directory for the calibration results
    """

    calib_ans_dir = Path("tests/images/cam_calib_charuco_images/calibration_answer")
    assert (
        calib_ans_dir.exists()
    ), f"calibration_answer_dir {calib_ans_dir} does not exist"
    return calib_ans_dir


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
    assert calib_res is None, (
        f"Failed on empty detections - calibration_result should be None, "
        f"got: {calib_res}"
    )


def test_calibrate_camera_happy_path(
    charuco_board: CharucoBoard,
    detections: Iterable[CharucoBoardDetection],
    calibration_image_paths: Iterable[Path],
    calibration_answer: CameraParameter,
):
    # Arrange

    # Take the first image to get the image size
    example_img = cv2.imread(str(next(iter(calibration_image_paths))))
    image_size = example_img.shape[:2]  # TODO: could be a fixture

    calib_res_dir = Path("tests/images/cam_calib_charuco_images/calibration_result")
    calib_res_dir.mkdir(exist_ok=True)
    calib_ans = calibration_answer

    # Act
    calib_res = charuco_board.calibrate_camera(image_size, detections)

    # Assert
    assert (
        calib_res is not None
    ), "Failed on happy path - calibration_result should not be None"
    calib_res.save_to(calib_res_dir / "calibration_result.npz")

    intrinsic_mat_same_shape = calib_res.K.shape == calib_ans.K.shape
    intrinsic_mat_same_value = np.allclose(
        calib_res.K.reshape(-1), calib_ans.K.reshape(-1)
    )

    assert intrinsic_mat_same_shape and intrinsic_mat_same_value, (
        "Incorrect intrinsic matrix, "
        f"same shape: {intrinsic_mat_same_shape}, "
        f"same value: {intrinsic_mat_same_value}\n"
        f"got:\n{calib_res.K}\nexpect:\n{calib_ans.K}\n"
    )

    distortion_coeffs_same_shape = (
        calib_res.distortion_coeffs.shape == calib_ans.distortion_coeffs.shape
    )
    distortion_coeffs_same_value = np.allclose(
        calib_res.distortion_coeffs.reshape(-1), calib_ans.distortion_coeffs.reshape(-1)
    )

    assert distortion_coeffs_same_shape and distortion_coeffs_same_value, (
        "Incorrect distortion coefficients, "
        f"same shape: {distortion_coeffs_same_shape}, "
        f"same value: {distortion_coeffs_same_value}\n"
        f"got:\n{calib_res.distortion_coeffs}\nexpect:\n{calib_ans.distortion_coeffs}\n"
    )
