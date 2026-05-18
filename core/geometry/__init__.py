"""
Geometry utilities: pure functions operating on geometric entities.

All functions in this module are stateless and side-effect-free.
Coordinate frames are explicitly documented for each function.
"""

from .projection import project_to_pixel
from .backprojection import pixel_to_camera_ray, pixel_to_world_ray
from .transforms import transform_point_world_to_camera, transform_point_camera_to_world
from .transforms import transform_ray_camera_to_world
from .reprojection import reprojection_error

__all__ = [
    "project_to_pixel",
    "pixel_to_camera_ray",
    "pixel_to_world_ray",
    "transform_point_world_to_camera",
    "transform_point_camera_to_world",
    "transform_ray_camera_to_world",
    "reprojection_error",
]
