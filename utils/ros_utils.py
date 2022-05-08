# Some of the code is from: https://github.com/eric-wieser/ros_numpy

import numpy as np

from sensor_msgs.msg import PointCloud2, PointField


DUMMY_FIELD_PREFIX = "__"
type_mappings = [
    (PointField.INT8, np.dtype("int8")),
    (PointField.UINT8, np.dtype("uint8")),
    (PointField.INT16, np.dtype("int16")),
    (PointField.UINT16, np.dtype("uint16")),
    (PointField.INT32, np.dtype("int32")),
    (PointField.UINT32, np.dtype("uint32")),
    (PointField.FLOAT32, np.dtype("float32")),
    (PointField.FLOAT64, np.dtype("float64")),
]
pftype_to_nptype = dict(type_mappings)
nptype_to_pftype = dict((nptype, pftype) for pftype, nptype in type_mappings)

# sizes (in bytes) of PointField types
pftype_sizes = {
    PointField.INT8: 1,
    PointField.UINT8: 1,
    PointField.INT16: 2,
    PointField.UINT16: 2,
    PointField.INT32: 4,
    PointField.UINT32: 4,
    PointField.FLOAT32: 4,
    PointField.FLOAT64: 8,
}


def fields_to_dtype(fields, point_step):
    """Convert a list of PointFields to a numpy record datatype."""
    offset = 0
    np_dtype_list = []
    for f in fields:
        while offset < f.offset:
            # might be extra padding between fields
            np_dtype_list.append(("%s%d" % (DUMMY_FIELD_PREFIX, offset), np.uint8))
            offset += 1

        dtype = pftype_to_nptype[f.datatype]
        if f.count != 1:
            dtype = np.dtype((dtype, f.count))

        np_dtype_list.append((f.name, dtype))
        offset += pftype_sizes[f.datatype] * f.count

    # might be extra padding between points
    while offset < point_step:
        np_dtype_list.append(("%s%d" % (DUMMY_FIELD_PREFIX, offset), np.uint8))
        offset += 1

    return np_dtype_list


def split_rgb_field(cloud_arr):
    """Takes an array with a named 'rgb' float32 field, and returns an array in which
    this has been split into 3 uint 8 fields: 'r', 'g', and 'b'.
    (pcl stores rgb in packed 32 bit floats)
    """
    rgb_arr = cloud_arr["rgb"].copy()
    rgb_arr.dtype = np.uint32
    r = np.asarray((rgb_arr >> 16) & 255, dtype=np.uint8)
    g = np.asarray((rgb_arr >> 8) & 255, dtype=np.uint8)
    b = np.asarray(rgb_arr & 255, dtype=np.uint8)

    # create a new array, without rgb, but with r, g, and b fields
    new_dtype = []
    for field_name in cloud_arr.dtype.names:
        field_type, field_offset = cloud_arr.dtype.fields[field_name]
        if not field_name == "rgb":
            new_dtype.append((field_name, field_type))
    new_dtype.append(("r", np.uint8))
    new_dtype.append(("g", np.uint8))
    new_dtype.append(("b", np.uint8))
    new_cloud_arr = np.zeros(cloud_arr.shape, new_dtype)

    # fill in the new array
    for field_name in new_cloud_arr.dtype.names:
        if field_name == "r":
            new_cloud_arr[field_name] = r
        elif field_name == "g":
            new_cloud_arr[field_name] = g
        elif field_name == "b":
            new_cloud_arr[field_name] = b
        else:
            new_cloud_arr[field_name] = cloud_arr[field_name]
    return new_cloud_arr


def pointcloud2_to_array(cloud_msg, squeeze=True):
    """Converts a rospy PointCloud2 message to a numpy recordarray

    Reshapes the returned array to have shape (height, width), even if the height is 1.
    The reason for using np.frombuffer rather than struct.unpack is speed... especially
    for large point clouds, this will be <much> faster.
    """
    # construct a numpy record type equivalent to the point type of this cloud
    dtype_list = fields_to_dtype(cloud_msg.fields, cloud_msg.point_step)

    # parse the cloud into an array
    cloud_arr = np.frombuffer(cloud_msg.data, dtype_list)

    # remove the dummy fields that were added
    cloud_arr = cloud_arr[
        [
            fname
            for fname, _type in dtype_list
            if not (fname[: len(DUMMY_FIELD_PREFIX)] == DUMMY_FIELD_PREFIX)
        ]
    ]

    if squeeze and cloud_msg.height == 1:
        return np.reshape(cloud_arr, (cloud_msg.width,))
    else:
        return np.reshape(cloud_arr, (cloud_msg.height, cloud_msg.width))


def get_xyz_points(cloud_array, remove_nans=True, dtype=np.float):
    # remove crap points
    if remove_nans:
        mask = (
            np.isfinite(cloud_array["x"])
            & np.isfinite(cloud_array["y"])
            & np.isfinite(cloud_array["z"])
        )
        cloud_array = cloud_array[mask]

    # pull out x, y, and z values
    points = np.zeros(cloud_array.shape + (3,), dtype=dtype)
    points[..., 0] = cloud_array["x"]
    points[..., 1] = cloud_array["y"]
    points[..., 2] = cloud_array["z"]

    return points


def get_points_and_colors(pointcloud: PointCloud2, remove_nans=True, dtype=np.float):
    cloud_array = pointcloud2_to_array(pointcloud)

    if remove_nans:
        mask = (
            np.isfinite(cloud_array["x"])
            & np.isfinite(cloud_array["y"])
            & np.isfinite(cloud_array["z"])
        )
        cloud_array = cloud_array[mask]

    cloud_array_rgb_splitted = split_rgb_field(cloud_array)

    # pull out x, y, and z
    points = np.zeros(cloud_array_rgb_splitted.shape + (3,), dtype=dtype)
    points[..., 0] = cloud_array_rgb_splitted["x"]
    points[..., 1] = cloud_array_rgb_splitted["y"]
    points[..., 2] = cloud_array_rgb_splitted["z"]

    # pull out rgb
    rgb = np.zeros(cloud_array_rgb_splitted.shape + (3,), dtype=dtype)
    rgb[..., 0] = cloud_array_rgb_splitted["r"]
    rgb[..., 1] = cloud_array_rgb_splitted["g"]
    rgb[..., 2] = cloud_array_rgb_splitted["b"]

    return points, rgb
