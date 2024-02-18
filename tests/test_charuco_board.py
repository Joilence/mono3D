"""Unit tests for the CharucoBoard class"""
from pathlib import Path
from typing import Any, Dict, Iterable

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from mono3d.charuco_board import CharucoBoard
from mono3d.charuco_board_detection import CharucoBoardDetection


@pytest.fixture
def charuco_board_image() -> npt.NDArray[np.uint8]:
    """
    Fixture to load the charuco board image
    """
    img = cv2.imread("tests/images/charuco_board.jpg")
    assert img is not None, "charuco_board_image should not be None"
    return img


@pytest.mark.parametrize(
    "test_id, configs",
    [
        (
                "happy_path_custom_dict",
                {"aruco_dict": cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250),  # Custom value
                 "n_horizontal_squares": 7,
                 "n_vertical_squares": 5,
                 "board_square_length": 60.0,
                 "aruco_marker_length": 45.0}
        ),
    ])
def test_CharucoBoard_init(test_id: str, configs: Dict[str, Any]):
    # Arrange
    def test_aruco_dict_equal(aruco_dict1, aruco_dict2):
        return (aruco_dict1.bytesList == aruco_dict2.bytesList).all()

    # Act
    board = CharucoBoard(**configs)

    # Assert
    assert board.n_horizontal_squares == configs["n_horizontal_squares"], f"Failed on {test_id} - n_horizontal_squares"
    assert board.n_vertical_squares == configs["n_vertical_squares"], f"Failed on {test_id} - n_vertical_squares"
    assert board.board_square_length == configs["board_square_length"], f"Failed on {test_id} - board_square_length"
    assert board.aruco_marker_length == configs["aruco_marker_length"], f"Failed on {test_id} - aruco_marker_length"
    assert test_aruco_dict_equal(board.aruco_dict, configs["aruco_dict"]), f"Failed on {test_id} - aruco_dict"


def test_detect_empty_image(charuco_board: CharucoBoard):
    # Arrange
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    # Act
    detection = charuco_board.detect(image)

    # Assert
    assert detection is None, f"detection should be None, got {detection}"


def test_detect_single_image(charuco_board: CharucoBoard, charuco_board_image: npt.NDArray[np.uint8]):
    # Arrange
    aruco_marker_ids = np.array([[20], [15], [10], [5], [18], [13],
                                 [8], [21], [16], [11], [9], [12],
                                 [0], [23], [3], [6], [1], [24],
                                 [19], [14], [22], [17], [7], [2]])

    charuco_ids = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8],
                            [9], [10], [11], [12], [13], [14], [15], [16],
                            [17], [18], [19], [20], [21], [22], [23], [24],
                            [25], [26], [27], [28], [29], [32], [33]])
    assert charuco_board_image is not None, "charuco_board_image should not be None"

    # Act
    detection = charuco_board.detect(charuco_board_image)

    # Assert
    assert isinstance(detection, CharucoBoardDetection), \
        f"detection should be CharucoBoardDetection, got {type(detection)}"
    assert aruco_marker_ids.shape == detection.aruco_marker_ids.shape, \
        f"aruco_marker_ids should have shape {aruco_marker_ids.shape} while got {detection.aruco_marker_ids.shape}"
    assert (detection.aruco_marker_ids == aruco_marker_ids).all(), \
        f"aruco_marker_ids should be:\n{aruco_marker_ids}\nwhile got:\n{detection.aruco_marker_ids}"
    assert (detection.charuco_ids == charuco_ids).all(), \
        f"charuco_ids should be:\n{charuco_ids}\nwhile got:\n{detection.charuco_ids}"


def test_detect_calibration_images(
        charuco_board: CharucoBoard,
        calibration_image_paths: Iterable[Path],
        detection_answer_dir: Path,
):
    # Arrange
    result_dir = Path("tests/images/cam_calib_charuco_images/detection_results")
    result_dir.mkdir(exist_ok=True)

    answer_dir = detection_answer_dir
    assert answer_dir.exists(), f"answer_dir {answer_dir} does not exist"

    for image_path in calibration_image_paths:
        image = cv2.imread(str(image_path))

        # Act
        detection = charuco_board.detect(image)

        # Assert
        assert detection is not None, f"detection should not be None for {image_path.stem}"
        detection.save_to(result_dir / f"{image_path.stem}.npz")
        det_ans = CharucoBoardDetection.load_from(answer_dir / f"{image_path.stem}.npz")
        assert detection.aruco_marker_ids.shape == det_ans.aruco_marker_ids.shape, \
            f"aruco_marker_ids in {image_path.stem} should have shape {det_ans.aruco_marker_ids.shape} while got {detection.aruco_marker_ids.shape}"
        assert detection.has_equal_ids(det_ans), \
            f"detection should be equal to detection_answer for {image_path.stem}, " \
            f"got:\n{detection.aruco_marker_ids}\nwhile answer is:\n{det_ans.aruco_marker_ids}"
