from typing import List, Optional
from matplotlib.pyplot import fill
import numpy as np
import gym
from gym import spaces
from omegaconf import OmegaConf
from robomimic.envs.env_robosuite import EnvRobosuite
import h5py
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import matplotlib.pyplot as plt
import robomimic.utils.obs_utils as ObsUtils
import robosuite.utils.transform_utils as T
import matplotlib.colors as mcolors

def get_camera_intrinsic_matrix(sim, camera_name, camera_height, camera_width):
    """
    Obtains camera intrinsic matrix.

    Args:
        sim (MjSim): simulator instance
        camera_name (str): name of camera
        camera_height (int): height of camera images in pixels
        camera_width (int): width of camera images in pixels
    Return:
        K (np.array): 3x3 camera matrix
    """
    cam_id = sim.model.camera_name2id(camera_name)
    fovy = sim.model.cam_fovy[cam_id]
    f = 0.5 * camera_height / np.tan(fovy * np.pi / 360)
    K = np.array([[f, 0, camera_width / 2], [0, f, camera_height / 2], [0, 0, 1]])
    return K

def get_camera_extrinsic_matrix(sim, camera_name):
    """
    Returns a 4x4 homogenous matrix corresponding to the camera pose in the
    world frame. MuJoCo has a weird convention for how it sets up the
    camera body axis, so we also apply a correction so that the x and y
    axis are along the camera view and the z axis points along the
    viewpoint.
    Normal camera convention: https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html

    Args:
        sim (MjSim): simulator instance
        camera_name (str): name of camera
    Return:
        R (np.array): 4x4 camera extrinsic matrix
    """
    cam_id = sim.model.camera_name2id(camera_name)
    camera_pos = sim.data.cam_xpos[cam_id]
    camera_rot = sim.data.cam_xmat[cam_id].reshape(3, 3)
    R = T.make_pose(camera_pos, camera_rot)

    # IMPORTANT! This is a correction so that the camera axis is set up along the viewpoint correctly.
    camera_axis_correction = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    R = R @ camera_axis_correction
    return R

def get_camera_transform_matrix(sim, camera_name, camera_height, camera_width):
    """
    Camera transform matrix to project from world coordinates to pixel coordinates.

    Args:
        sim (MjSim): simulator instance
        camera_name (str): name of camera
        camera_height (int): height of camera images in pixels
        camera_width (int): width of camera images in pixels
    Return:
        K (np.array): 4x4 camera matrix to project from world coordinates to pixel coordinates
    """
    R = get_camera_extrinsic_matrix(sim=sim, camera_name=camera_name)
    K = get_camera_intrinsic_matrix(
        sim=sim, camera_name=camera_name, camera_height=camera_height, camera_width=camera_width
    )
    K_exp = np.eye(4)
    K_exp[:3, :3] = K

    # Takes a point in world, transforms to camera frame, and then projects onto image plane.
    return K_exp @ T.pose_inv(R)

def get_real_camera_transform_matrix(extrinsic, intrinsic, camera_height, camera_width):
    K_exp = np.eye(4)
    K_exp[:3, :3] = intrinsic
    return K_exp @ T.pose_inv(extrinsic)

def project_points_from_world_to_camera(points, world_to_camera_transform, camera_height, camera_width):
    """
    Helper function to project a batch of points in the world frame
    into camera pixels using the world to camera transformation.

    Args:
        points (np.array): 3D points in world frame to project onto camera pixel locations. Should
            be shape [..., 3].
        world_to_camera_transform (np.array): 4x4 Tensor to go from robot coordinates to pixel
            coordinates.
        camera_height (int): height of the camera image
        camera_width (int): width of the camera image

    Return:
        pixels (np.array): projected pixel indices of shape [..., 2]
    """
    assert points.shape[-1] == 3  # last dimension must be 3D
    assert len(world_to_camera_transform.shape) == 2
    assert world_to_camera_transform.shape[0] == 4 and world_to_camera_transform.shape[1] == 4

    # convert points to homogenous coordinates -> (px, py, pz, 1)
    ones_pad = np.ones(points.shape[:-1] + (1,))
    points = np.concatenate((points, ones_pad), axis=-1)  # shape [..., 4]

    # batch matrix multiplication of 4 x 4 matrix and 4 x 1 vectors to do robot frame to pixels transform
    mat_reshape = [1] * len(points.shape[:-1]) + [4, 4]
    cam_trans = world_to_camera_transform.reshape(mat_reshape)  # shape [..., 4, 4]
    pixels = np.matmul(cam_trans, points[..., None])[..., 0]  # shape [..., 4]

    # re-scaling from homogenous coordinates to recover pixel values
    # (x, y, z) -> (x / z, y / z)
    pixels = pixels / pixels[..., 2:3]
    pixels = pixels[..., :2].round().astype(int)  # shape [..., 2]

    # swap first and second coordinates to get pixel indices that correspond to (height, width)
    # and also clip pixels that are out of range of the camera image
    pixels = np.concatenate(
        (
            pixels[..., 1:2].clip(0, camera_height - 1),
            pixels[..., 0:1].clip(0, camera_width - 1),
        ),
        axis=-1,
    )

    return pixels

def proj_points_real(point):
    # Assuming p_point_robot is the point in the robot base frame
    offset = np.array([-0.56, 0, 0.912])
    p_point_robot = point

    K = np.array([[637.42955067,   0.,         311.31550407],
    [  0.,         638.15245514, 262.01036327],
    [  0.,           0.,           1.        ]])

    cam2base = np.array([[-0.56336906,  0.09248256, -0.82101296,  1.28781094],
    [ 0.06481906,  0.9955999,   0.06767077,  0.01550116],
    [ 0.82365879, -0.01509367, -0.56688481,  0.57642152],
    [ 0.,          0.,          0.,          1.        ]])

    # # Reverse the offset
    # p_point_robot -= offset

    # # Reverse transformation to camera coordinates
    # p3_cam_homo_reverse = np.append(p_point_robot, 1.0)  # Convert to homogeneous coordinates
    # p3_cam_reverse = np.linalg.inv(cam2base) @ p3_cam_homo_reverse  # Reverse transformation to camera coordinates

    # # Convert to image plane
    # p2_homo_reverse = np.linalg.inv(K) @ p3_cam_reverse[:3]  # Transformation to image plane
    # u_reverse, v_reverse, _ = p2_homo_reverse / p3_cam_reverse[2]  # Normalize homogeneous coordinates
    # return u_reverse, v_reverse

    # Assuming p3_robot is the point in the robot base frame
    p3_robot = point - offset #np.array([x_robot, y_robot, z_robot])  # Replace with actual values

    # # Convert point to homogeneous coordinates
    # p3_robot_homo = np.hstack((p3_robot, 1))

    # # Calculate inverse of the transformation matrix
    # inv_cam2base = np.linalg.inv(cam2base)  # Assuming cam2base is available

    # # Transform point from robot base frame to camera frame
    # p3_cam_homo = inv_cam2base @ p3_robot_homo

    # # Convert point to Euclidean coordinates
    # p3_cam = p3_cam_homo[:-1] / p3_cam_homo[-1]

    # # Use intrinsic matrix K to recover image coordinates (u, v)
    # p2_homo = K @ p3_cam
    # u, v = p2_homo[:2] / p2_homo[2]

    # inv_cam2base = np.linalg.inv(cam2base)  # Assuming cam2base is available

    # # Transform point from robot base frame to camera frame
    # p3_cam = inv_cam2base.dot(np.append(p3_robot, 1))[:3]

    # # Use intrinsic matrix K to recover image coordinates (u, v)
    # p2_homo = K @ p3_cam
    # u, v = p2_homo[:2] / p2_homo[2]

    # Step 1: Inverse transformation from robot coordinates to camera coordinates
    inv_cam2base = np.linalg.inv(cam2base)
    p3_cam_homo = np.append(inv_cam2base @ np.append(p3_robot, [1]), 1)

    # Step 2: Convert from homogeneous coordinates to Cartesian coordinates
    p3_cam = p3_cam_homo[:3] / p3_cam_homo[3]

    # Step 3: Project the 3D point onto the image plane (without depth)
    p2_normalized = K @ p3_cam

    # Extract pixel coordinates (u, v)
    u = (p2_normalized[0] / p2_normalized[2])
    v = (p2_normalized[1] / p2_normalized[2])
    return u, v


import numpy as np

def get_real_extrinsic():
    # Assuming you have the camera intrinsics matrix K and cam2base matrix

    cam2base = np.array([[-0.56336906,  0.09248256, -0.82101296,  1.28781094],
    [ 0.06481906,  0.9955999,   0.06767077,  0.01550116],
    [ 0.82365879, -0.01509367, -0.56688481,  0.57642152],
    [ 0.,          0.,          0.,          1.        ]])

    # Extract rotation and translation components from cam2base matrix
    R_cam2base = cam2base[:3, :3]
    t_cam2base = cam2base[:3, 3]

    # Create the extrinsic matrix for the camera in the base frame
    extrinsics = np.vstack((np.hstack((R_cam2base, t_cam2base.reshape(-1, 1))), [0, 0, 0, 1]))

    return extrinsics
# dataset_path = '/srv/rl2-lab/flash8/mbronars3/RAL/datasets/new_block_reach_long.hdf5'

# with h5py.File(dataset_path, "r") as f:
#     states = f["data/demo_0/states"][()]
#     initial_state = dict(states=states[0])
#     initial_state["model"] = f["data/{}".format("demo_0")].attrs["model_file"]

# env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
# env_name = env_meta["env_name"]
# dummy_spec = dict(
#             obs=dict(
#                     low_dim=["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
#                     rgb=[],
#                 ),
#         )

# ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)
# env = EnvUtils.create_env_from_metadata(
#             env_meta=env_meta,
#             env_name=env_name, 
#             render_offscreen=True,
#         )

# env.reset()
# # env.reset_to(initial_state)
# img = env.render(mode="rgb_array", height=512, width=512, camera_name='agentview')

img_path = "/srv/rl2-lab/flash8/mbronars3/workspace/results/images/chessboard_43.png"

img = plt.imread(img_path)
print(img.shape)

# save image
# plt.imsave('/srv/rl2-lab/flash8/mbronars3/workspace/results/images/test.png', img)

# camera_transform = get_camera_transform_matrix(env.env.sim, 'agentview', 512, 512)
intrinsic = np.array([[637.42955067,   0.,         311.31550407],
    [  0.,         638.15245514, 262.01036327],
    [  0.,           0.,           1.        ]])
camera_transform = get_real_camera_transform_matrix(get_real_extrinsic(), intrinsic, 480, 640)

# points_path = "/srv/rl2-lab/flash8/mbronars3/RAL/block_reach/sim/diffusion_policy/ablations/a0.8_g0.75_w10/gamma/0.5/w10/trajs.hdf5"
# points_path = '/srv/rl2-lab/flash8/mbronars3/RAL/block_reach/sim/diffusion_policy/param_tuning/11.12_e125/vanillaDP/trajs.hdf5'
# points_path = '/srv/rl2-lab/flash8/mbronars3/RAL/block_reach/sim/BeT/eval/training_seed/trajs.hdf5'
# points_path = '/srv/rl2-lab/flash8/mbronars3/RAL/block_reach/sim/IBC/eval/e700_2/trajs.hdf5'
points_path = '/srv/rl2-lab/flash8/mbronars3/RAL/datasets/calibrate_real.hdf5'
# points = []
# max_length = 0
plt.imshow(img)
with h5py.File(points_path, "r") as f:
    # for demo in f["data"]:
    #     _, length, _, _ = f['data'][demo]['next_obs'].shape
    #     # max_length = max(max_length, length)
    #     # demo_points = []
    #     for i in range(length):
    #         points = f['data'][demo]['next_obs'][:, i, 1, [20, 21, 22]]
    #         red = f['data'][demo]['next_obs'][:, i, 1, 2] > 0.84
    #         green = f['data'][demo]['next_obs'][:, i, 1, 9] > 0.84
    #         # red true if any value in the array is greater than 0.84
    #         # get first index where red is true
    #         first_green = np.argmax(green)
    #         first_red = np.argmax(red)
    #         red = np.any(red)
            
    #         # plot line through list of points
    #         # make it so the color goes from blue to orange as the points get closer to the end
    #         if red:
    #             colors = [(0, "lightcoral"), (1, "darkblue")]
    #             proj_p = project_points_from_world_to_camera(points, camera_transform, 512, 512)[:first_red]
    #         else:
    #             colors = [(0, "palegreen"), (1, "darkblue")]
    #             proj_p = project_points_from_world_to_camera(points, camera_transform, 512, 512)[:first_green]
    #         cmap = mcolors.LinearSegmentedColormap.from_list("mycmap", colors)
    #         for j in range(len(proj_p) - 1):
    #             plt.plot(proj_p[j:j+2, 1], proj_p[j:j+2, 0], c=cmap(j/len(proj_p)), linewidth=2.5)
    #         # plt.plot(proj_p[:, 1], proj_p[:, 0], c=plt.cm.coolwarm(i/length))

    for demo in f["data"]:
        # points = f['data'][demo]['next_obs'][:, [20, 21, 22]]
        # red = f['data'][demo]['next_obs'][:, 2] > 0.84
        # green = f['data'][demo]['next_obs'][:, 9] > 0.84
        points = f['data'][demo]['obs']['robot0_eef_pos'][()]
        # subtract .16 from x value of each point
        points[:, 0] += .164
        points[:, 1] += .00265
        points[:, 2] -= 1.5 #.983

        

        final_heights = f['data'][demo]['obs']['object'][:, [2, 9]][-1]
        red = f['data'][demo]['obs']['object'][:, 2] > 0.98
        green = f['data'][demo]['obs']['object'][:, 9] > 0.98
        # red true if any value in the array is greater than 0.84
        # get first index where red is true
        first_green = np.argmax(green)
        first_red = np.argmax(red)
        red = np.any(red)
        green = np.any(green)
        if not red and not green:
            continue
        
        # plot line through list of points
        # make it so the color goes from blue to orange as the points get closer to the end
        if red:
            colors = [(0, "lightcoral"), (1, "darkblue")]
            # points = np.array([0.0, 0.0, 0.0])
            proj_p = proj_points_real(points)
            # proj_p = project_points_from_world_to_camera(points, camera_transform, 480, 640)[:first_red]
        else:
            colors = [(0, "palegreen"), (1, "darkblue")]
            # points = np.array([0.24, 0.004, 0.048])
            proj_p = proj_points_real(points)
            
            # proj_p = project_points_from_world_to_camera(points, camera_transform, 480, 640)[:first_green]
        # from IPython import embed; embed()
        # from IPython import embed; embed()
        cmap = mcolors.LinearSegmentedColormap.from_list("mycmap", colors)
        # plt.scatter(proj_p[0], proj_p[1], s=5)
        for j in range(len(proj_p) - 1):
            plt.plot(proj_p[j:j+2, 1], proj_p[j:j+2, 0], c=cmap(j/len(proj_p)), linewidth=2.5)
        # plt.plot(proj_p[:, 1], proj_p[:, 0], c=plt.cm.coolwarm(i/length))

            


            # demo_points.append(point)
        # points.append(demo_points)

# Pad shorter demos with zeros and stack
# points = [np.pad(demo_points, ((0, max_length - len(demo_points)), (0, 0), (0, 0))) for demo_points in points]
# points = np.stack(points)

# from IPython import embed; embed()

# projected_points = project_points_from_world_to_camera(points, camera_transform, 512, 512)

# overlay the points on the image and save, dont show image
# plt.imshow(img)
# plt.scatter(projected_points[:, 1], projected_points[:, 0], s=1, c='r')
plt.axis('off')
plt.savefig('/srv/rl2-lab/flash8/mbronars3/workspace/results/images/test_points.png', bbox_inches='tight', pad_inches=0)

# from IPython import embed; embed()