import numpy as np
from scipy.interpolate import RegularGridInterpolator

from analyze_3d_mesh_reference_shift import tensor_field


def test_tensor_field_reconstructs_linear_temperature_at_irregular_order():
    x = np.array([0.0, 0.5, 1.0])
    y = np.array([0.0, 0.25, 1.0])
    z = np.array([0.0, 1.0])
    points = np.array(np.meshgrid(x, y, z, indexing="ij")).reshape(3, -1)
    values = 4 * points[0] + 2 * points[1] - points[2] + 10
    order = np.arange(points.shape[1])[::-1]
    axes, field = tensor_field(points[:, order], values[order])
    probe = np.array([[0.3, 0.7, 0.6]])
    assert np.allclose(RegularGridInterpolator(axes, field)(probe), 12.0)
