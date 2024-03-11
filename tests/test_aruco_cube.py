"""Unit tests for the ArucoCube class"""

import cv2
import cv2.typing as cvt
import pytest

from mono3d.aruco_cube import ArucoCube


@pytest.fixture
def aruco_cube_image() -> cvt.MatLike:
    """
    Fixture to load the charuco board image
    """
    img = cv2.imread("tests/images/aruco_cube.jpg")
    assert img is not None, "aruco_cube_image should not be None"
    return img


def test_detect_markers(aruco_cube_image):
    # Arrange
    aruco_cube = ArucoCube()

    # Act
    markers = aruco_cube.detect(aruco_cube_image)

    # Assert
    assert markers is not None, "markers should not be None"
    assert len(markers) == 3, f"markers should have length 3, got {len(markers)}"
    # TODO: check IDs
    # TODO: check coords
