import numpy as np
import cv2


EPS = 1e-8


def backproject_corners(K, width, height, depth, Rt):
    """
    Backproject 4 corner points in image plane to the imaginary depth plane using intrinsics and extrinsics
    
    Hint:
    depth * corners = K @ T @ y, where y is the output world coordinates and T is the 4x4 matrix of Rt (3x4)

    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        width -- width of the image
        heigh -- height of the image
        depth -- depth value of the imaginary plane
    Output:
        points -- 2 x 2 x 3 array of 3D coordinates of backprojected points, here 2x2 correspinds to 4 corners
    """

    points = np.array([[0, 0,1], [width, 0,1], [0, height,1], [width, height,1]])
 
    
    p=np.linalg.inv(K) @ points.T
    p=p/p[-1,:]
    p=depth*p
    point = np.linalg.inv(Rt[:3,:3]) @ (p - Rt[:,-1].reshape(-1,1))
    
    return (point.T).reshape(2,2,3)


def project_points(K, Rt, points):
    """
    Project 3D points into a calibrated camera.
    
    Hint:
    Z * projections = K @ T @ p, where p is the input points and projections is the output, T is the 4x4 matrix of Rt (3x4)
    
    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- points_height x points_width x 3 array of 3D points
    Output:
        projections -- points_height x points_width x 2 array of 2D projections
    """
    """ YOUR CODE HERE
    """
    h=points.shape[0]
    w=points.shape[1]
    
    points = points.reshape((h*w, 3))

    point = (Rt[:3,:3] @ points.T) + Rt[:,-1].reshape(3,1)
    points = np.hstack((points, np.ones((h*w,1))))

    projection=K@point
    projection /= projection[-1,:].reshape(1,-1)
    projection = projection

    return projection[:-1,:].T.reshape((h,w,2))




def warp_neighbor_to_ref(
    backproject_fn, project_fn, depth, neighbor_rgb, K_ref, Rt_ref, K_neighbor, Rt_neighbor
):
    """
    Warp the neighbor view into the reference view
    via the homography induced by the imaginary depth plane at the specified depth

    Make use of the functions you've implemented in the same file (which are passed in as arguments):
    - backproject_corners
    - project_points

    Also make use of the cv2 functions:
    - cv2.findHomography
    - cv2.warpPerspective
    
    ! Note, when you use cv2.warpPerspective, you should use the shape (width, height), NOT (height, width)
    
    Hint: you should do the follows:
    1.) apply backproject_corners on ref view to get the virtual 3D corner points in the virtual plane
    2.) apply project_fn to project these virtual 3D corner points back to ref and neighbor views
    3.) use findHomography to get teh H between neighbor and ref
    4.) warp the neighbor view into the reference view

    Input:
        backproject_fn -- backproject_corners function
        project_fn -- project_points function
        depth -- scalar value of the depth at the imaginary depth plane
        neighbor_rgb -- height x width x 3 array of neighbor rgb image
        K_ref -- 3 x 3 camera intrinsics calibration matrix of reference view
        Rt_ref -- 3 x 4 camera extrinsics calibration matrix of reference view
        K_neighbor -- 3 x 3 camera intrinsics calibration matrix of neighbor view
        Rt_neighbor -- 3 x 4 camera extrinsics calibration matrix of neighbor view
    Output:
        warped_neighbor -- height x width x 3 array of the warped neighbor RGB image
    """

    height, width = neighbor_rgb.shape[:2]
    

    """ YOUR CODE HERE
    """
    points = np.array([[0, 0,1], [width, 0,1], [0, height,1], [width, height,1]])
    p_i = backproject_fn(K_ref, width, height, depth, Rt_ref)
    p_j = project_fn(K_neighbor, Rt_neighbor, p_i)
    pp_j=p_j.reshape((p_j.shape[0]*p_j.shape[1],2))
    H, mask = cv2.findHomography(points, pp_j, cv2.RANSAC)
    K_inv=np.linalg.inv(H)
    warped_neighbor = cv2.warpPerspective(neighbor_rgb, K_inv, dsize =(width, height))
    """ END YOUR CODE
    """
    return warped_neighbor


def zncc_kernel_2D(src, dst):
    """
    Compute the cost map between src and dst patchified images via the ZNCC metric

    IMPORTANT NOTES:
    - Treat each RGB channel separately but sum the 3 different zncc scores at each pixel

    - When normalizing by the standard deviation, add the provided small epsilon value,
    EPS which is included in this file, to both sigma_src and sigma_dst to avoid divide-by-zero issues

    Input:
        src -- height x width x K**2 x 3
        dst -- height x width x K**2 x 3
    Output:
        zncc -- height x width array of zncc metric computed at each pixel
    """
    assert src.ndim == 4 and dst.ndim == 4
    assert src.shape[:] == dst.shape[:]

    """ YOUR CODE HERE
    """
    src_mean = (np.mean(src, 2))[:,:,np.newaxis, :]
    dst_mean = (np.mean(dst, 2))[:,:,np.newaxis, :]
    Wi_sigma=np.std(src, 2)
    Wj_sigma = np.std(dst, 2)


    zncc = np.sum(np.sum(np.subtract(src, src_mean) * np.subtract(dst,dst_mean), 2) / (Wi_sigma * Wj_sigma + EPS),2)


    """ END YOUR CODE
    """

    return zncc 


def backproject(dep_map, K):
    """
    Backproject image points to 3D coordinates wrt the camera frame according to the depth map

    Input:
        K -- camera intrinsics calibration matrix
        dep_map -- height x width array of depth values
    Output:
        points -- height x width x 3 array of 3D coordinates of backprojected points
    """
    _u, _v = np.meshgrid(np.arange(dep_map.shape[1]), np.arange(dep_map.shape[0]))

    """ YOUR CODE HERE
    """
    pcl=np.zeros((dep_map.shape[0],dep_map.shape[1],3), dtype=np.float64)
    for x in range(dep_map.shape[0]):
        for y in range(dep_map.shape[1]):
            pcl[x,y,0]=(y-K[0,2])*dep_map[x,y]/K[0,0]
            pcl[x,y,1]=(x-K[1,2])*dep_map[x,y]/K[1,1]
            pcl[x,y,2]=dep_map[x,y]

    xyz_cam = pcl
    """ END YOUR CODE
    """
    return xyz_cam
