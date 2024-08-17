# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
from unittest import TestCase

import numpy as np
import torch as th
from arvr.projects.facetracking.hmd.pythonbindings.engine import camera_operator

# @manual=fbsource//arvr/projects/codec_avatar/drtk:utils
from drtk.utils.projection import project_fisheye_distort_62


def get_meshgrid_vec(width: int, height: int) -> np.ndarray:
    image_indices = np.meshgrid(np.arange(width), np.arange(height))
    image_indices = np.stack(image_indices, axis=-1)
    return image_indices.reshape(-1, 2).astype(np.float32)


def generate_unit_plane(
    camera: camera_operator.CameraOperator, width: int, height: int
) -> np.ndarray:
    image_indices_vec = get_meshgrid_vec(width, height)

    uv_positions_2d = camera.unproject_unit_plane(
        image_indices_vec, apply_undistortion=False
    ).astype(np.float64)

    uv_positions_3d = np.concatenate(
        [
            uv_positions_2d,
            np.ones([image_indices_vec.shape[0], 1], dtype=uv_positions_2d.dtype),
        ],
        axis=1,
    )
    return uv_positions_3d


class PerceptionParityTest(TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_fisheye62(self) -> None:
        # Hardcoded fisheye62 with no flips and no LUT
        n_distortion_parameters = 8
        width = 100
        height = 200
        random_intrinsics = np.eye(4, dtype=np.float64)
        random_intrinsics[0, 0] = np.random.uniform(low=0.9, high=1.0) * max(
            height, width
        )
        random_intrinsics[1, 1] = random_intrinsics[0, 0]
        random_intrinsics[0, 2] = np.random.uniform(low=0.4, high=0.6) * width
        random_intrinsics[1, 2] = np.random.uniform(low=0.4, high=0.6) * height
        distortion = np.random.uniform(low=-0.1, high=0.1, size=n_distortion_parameters)

        # Crate the perception fisheye62 camera
        fisheye = camera_operator.CameraOperator.init_with_fisheye(
            extrinsic_matrix=np.eye(4).astype(np.float64),
            intrinsic_matrix=random_intrinsics,
            distortion=distortion,
            width=width,
            height=height,
            name="test",
            id=12,
        )
        self.assertEqual(fisheye.model_type, "FISHEYE62_WITH_BOTH_FOCAL")

        # Create the unprojected points using the perception camera (these are
        # effectively arbitrary)
        unit_plane_points = generate_unit_plane(fisheye, width, height)

        perception_output = fisheye.apply_intrinsic(
            unit_plane_points, apply_distortion=True
        )

        # Compare with drtk
        focal = th.tensor(fisheye.intrinsic_matrix[:2, :2], dtype=th.float)
        drtk_output = project_fisheye_distort_62(
            th.tensor(unit_plane_points, dtype=th.float).unsqueeze(0),
            th.diag_embed(th.diagonal(focal)).unsqueeze(0),
            th.tensor(
                [
                    fisheye.principal_point_x,
                    fisheye.principal_point_y,
                ],
                dtype=th.float,
            ).unsqueeze(0),
            th.tensor(fisheye.distortion, dtype=th.float).unsqueeze(0),
        )

        np.testing.assert_allclose(
            perception_output, drtk_output.squeeze(0).numpy(), atol=1e-4, rtol=1e-4
        )
