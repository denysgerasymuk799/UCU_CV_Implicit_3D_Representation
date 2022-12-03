import numpy as np

from mesh_to_sdf.utils import scale_to_unit_sphere, sample_uniform_points_in_unit_sphere
from mesh_to_sdf import get_surface_point_cloud
from mesh_to_sdf.surface_point_cloud import BadMeshException


def sample_sdf_random_points(mesh, number_of_points=500000, surface_point_method='scan', sign_method='normal',
                             scan_count=100, scan_resolution=400, sample_point_count=10000000, normal_sample_count=11,
                             min_size=0, return_gradients=False):
    mesh = scale_to_unit_sphere(mesh)

    if surface_point_method == 'sample' and sign_method == 'depth':
        print("Incompatible methods for sampling points and determining sign, using sign_method='normal' instead.")
        sign_method = 'normal'

    surface_point_cloud = get_surface_point_cloud(mesh, surface_point_method, 1, scan_count, scan_resolution,
                                                  sample_point_count,
                                                  calculate_normals=sign_method == 'normal' or return_gradients)

    return generate_sdf_near_surface(surface_point_cloud, number_of_points, surface_point_method == 'scan', sign_method,
                                     normal_sample_count, min_size, return_gradients)


def generate_sdf_near_surface(surface_point_cloud, number_of_points=500000, use_scans=True, sign_method='normal',
                              normal_sample_count=11, min_size=0, return_gradients=False):
    query_points = []
    unit_sphere_sample_count = number_of_points
    unit_sphere_points = sample_uniform_points_in_unit_sphere(unit_sphere_sample_count)
    query_points.append(unit_sphere_points)
    query_points = np.concatenate(query_points).astype(np.float32)

    if sign_method == 'normal':
        sdf = surface_point_cloud.get_sdf_in_batches(query_points, use_depth_buffer=False, sample_count=normal_sample_count,
                                                     return_gradients=return_gradients)
    elif sign_method == 'depth':
        sdf = surface_point_cloud.get_sdf_in_batches(query_points, use_depth_buffer=True, return_gradients=return_gradients)
    else:
        raise ValueError('Unknown sign determination method: {:s}'.format(sign_method))
    if return_gradients:
        sdf, gradients = sdf

    if min_size > 0:
        model_size = np.count_nonzero(sdf[-unit_sphere_sample_count:] < 0) / unit_sphere_sample_count
        if model_size < min_size:
            raise BadMeshException()

    return query_points, sdf
