import cv2
from packaging import version

LEGACY = (version.parse(cv2.__version__)) < version.parse("4.6.0")

