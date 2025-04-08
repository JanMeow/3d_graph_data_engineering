
import numpy as np
import geometry_processing as GP
import collision as C
from scipy.optimize import linear_sum_assignment
from collections import Counter 

# Compute the apropriate type of the element based on the geometry and the position of the element
# ====================================================================
"""
    ### Two step classification apporach
    clasfication label 
    Computes the most possible element type of an ifcProxy element.
    or later extend to seeing if elements are mistyped based on a confidence score
    Below are some observations to generate a criterion:
    1.0 Relative position to the horizontal neighbours
        1.0.1 if the element is between two walls horizontally (approximate), it has a high chance being a wall
        1.0.2 if the element is between two slabs horizontally (approximate), it has a high chance being a slab
    1.1 Relative position to the vertical neighbours
        1.1.1 if the element is between two walls vertically (approximate), it has a high chance being a slab
        1.1.2 if the element is between two slabs vertically (approximate), it has a high chance being a wall
        1.1.3 if the element is between a roof and a slab vertically (approximate), it has a high chance being a wall or a column
        1.1.4 if the element is between two slab vertically (approximate), it has a high chance being a wall or a column
    1.2 Number of neighbourss
        1.2.1 if the element has two horizontal neighbours, it has a high chance being a wall
        1.2.2 if the element has two vertical neighbours, upper being a roof and lower being a slab, it has a high chance being a wall or column
    2. roof: angled, highest in the relative position to the building
    3. slab: flat base area
    4. roof: flat base area, but tilted
    ...
    ...
    ...
    Many of the above observations could be derived to help deduce the nature of an element.
    As such, this module works to formulate an algoirthm to compute the most probable type of an element and apply ML algorithm based on the criterion above.

    We first need to calculate the following features:
    1. Get its upper, lower, left, right neighbours type fill NaN if the neighbour is also proxy, and if no upper neighbout (in the case of roof)
        => assign_neighbours(node)
    2. Get its relative position to the building and to the floor (this might require you to group elements by floor and then by building
    2  as they might not be specifid in the model or the model has a lot of noises e.g, unwanted geometries eg funtirures...)
    2. Whether it is tilted and, maybe how tilted it is.
    3. Likelihood. for example, when we model, we model often multiple walls touching each other, and slabs, so we need to calculate also the 
        their tendecy of clustering.
    4.Proportion of the base curve => determining if its horiozontal or vertical or slanted element => determind_oobb_proportion(node)
    """
# ===================================================================================
# Global Variables for features type and rounding
# ===================================================================================
classfication_label = {
    "IfcWall": 0, #Can further classify into outer or inner wall
    "IfcSlab": 1,
    "IfcRoof": 2,
    "IfcColumn": 3,
    "IfcBeam": 4,
    "IfcCurtainWall": 5,
    "IfcFooting": 6,
}
Intrinsic_features= {
    "AABB_X_Extent": float,
    "OOBB_X_Extent": float,
    "AABB_Y_Extent": float,
    "OOBB_Y_Extent": float,
    "AABB_Z_Extent": float,
    "OOBB_Z_Extent": float,
    "AABB_base_area": float,
    "OOBB_Base_area": float,
    "World_CP": float,
    "World_X_start": float,
    "World_X_end": float,
    "World_Y_start": float,
    "World_Y_end": float,
    "World_Z_start": float,
    "World_Z_end": float,
    "self_x_start":float,
    "self_x_end":float,
    "self_y_start":float,
    "self_y_end":float,
    "self_z_start":float,
    "z_axis_aligned": bool,
    "number_of_vertices_in_base": int,
    "total_number_of_vertices": int,
    "total_number_of_faces": int,
    # ==================================
    # Can kind of approximate this by extending the bounding box volume by typical floor height
    # and then computing the relative position to the building?
    "relative_position_to_storey": float, #problem could be some ifc models dont even have building, are there alternatives?
    # We assume that we have one building for now, if not can run some clusterning alogoithm to determine how many buildings are there ?
    "relative_position_to_building": float, #problem could be some ifc models dont even have building, are there alternatives?
}
Contextural_features = {
    "upper": "ifctype",
    "lower": "ifctype",
    "left": "ifctype",
    "right": "ifctype",
    "number_of_neighbours_of_same_type": int,
    "variances of the direct neighbours cp": float, # not sure
    "cluster_size":int,
    "cluster_cp_distribution": float, 
    "horizontal_relatives": list[str("ifctype")], # python counter of dict varying vector length though and positon information is important
    "vertical_relatives": list[str("ifctype")],
}
cluster_features ={
    "cluster_size": 12,
    "mean_height": 3.0,
    "std_height": 0.2,
    "orientation_variance": 0.05,
    "bbox_volume": 86.0,
    "centroid_x": 42.3,
    "centroid_y": 15.2,
    "z_alignment_consistency": 1.0
}
round_to = 2
# ===================================================================================
# ===================================================================================
# ====================================================================
# Compute the features for the element
# ====================================================================
def get_Intrinsic_features(graph, guid):
    """
    The algorithm should also work a bit differently depending on whether there is labelled element
    if there is none, we need to first guess the element not based on the surrounding elements but based on the element itself
    then do it iteratively until it gives the highest confidence score

    *Optimization*:
    Currently uses a O(n) face normal check for otpmization to know if we can skip OOBB check which is O3 
    In the future, can do a sampling for the first 20 meshes to see if over 50% of them are axis aligned
    if yes, then keep the hybrid apporach, if not, then just ignore the check axis aligned and perform PCA directly
    another way is to use the is_xyz_aligned function to check which axis is aligned and then reduce the PCA n_component 

    """
    node = graph[guid]
    world_xyz = graph.bbox
    self_xyz = node.geom_info["bbox"]
    # Check if object is XYZ Axis aligned, if yes no need check PCA 
    is_xyz_aligned = is_axis_aligned(node, atol = 1e-3, threshold = 0.9)
    number_of_vertices_in_base, AABB_base_area = get_base_info(node)
    # PCA if needed
    if is_xyz_aligned:
        node.principal_axes = np.eye(3)
        node.is_axis_aligned = True
        node.z_axis_aligned = True
        min_max_extents = self_xyz[1] - self_xyz[0]
        OOBB_base_area = AABB_base_area
    else:
        node.is_axis_aligned = False
        principal_axes, min_max_extents= get_oobb(node)
        node.principal_axes = principal_axes
        node.z_axis_aligned = is_z_axis_aligned(node, atol = 1e-2)
        # Get the base vertex number and area
        OOBB_base_area = np.around(min_max_extents[0] * min_max_extents[1], round_to)
    # Features 
    Intrinsic_features= {
    "AABB_X_Extent": node.geom_info["bbox"][1][0] - node.geom_info["bbox"][0][0],
    "OOBB_X_Extent": min_max_extents[0],
    "AABB_Y_Extent": node.geom_info["bbox"][1][1] - node.geom_info["bbox"][0][1],
    "OOBB_Y_Extent": min_max_extents[1],
    "AABB_Z_Extent": node.geom_info["bbox"][1][2] - node.geom_info["bbox"][0][2],
    "OOBB_Z_Extent": min_max_extents[2],
    "AABB_base_area": AABB_base_area,
    "OOBB_Base_area": OOBB_base_area,
    "World_CP": GP.get_centre_point(world_xyz),
    "World_X_start": world_xyz[0][0],
    "World_X_end": world_xyz[1][0],
    "World_Y_start": world_xyz[0][1],
    "World_Y_end": world_xyz[1][1],
    "World_Z_start": world_xyz[0][2],
    "World_Z_end": world_xyz[1][2],
    "self_x_start":self_xyz[0][0],
    "self_x_end":self_xyz[1][0],
    "self_y_start":self_xyz[0][1],
    "self_y_end":self_xyz[1][1],
    "self_z_start":self_xyz[0][2],
    "self_z_end":self_xyz[1][2],
    "z_axis_aligned": node.z_axis_aligned,
    "number_of_vertices_in_base": number_of_vertices_in_base,
    "total_number_of_vertices": len(node.geom_info["vertex"]),
    "total_number_of_faces": len(node.geom_info["face"])
    }
    node.intrinsic_features = Intrinsic_features
    return Intrinsic_features
def get_contextural_features(graph, guid):
    node = graph[guid]
    self_cp = GP.get_centre_point(node.geom_info["bbox"])
    # Get Neighbours
    neighbours = assign_neighbours(node)
    # Get horizontal neighbours and their counts
    horizontal_relatives = get_horizontal_relatives(graph, node)
    # Get Vertical neighbours
    vertical_relatives = None
    # Get Cluster size and distribution
    cluster = get_cluster(node)
    Cluster_features = get_cluster_features(cluster)
     # Result
    Contextural_features = {
    "upper": neighbours[0],
    "lower": neighbours[1],
    "left": neighbours[2],
    "right": neighbours[3],
    "number_of_neighbours_of_same_type": len([n for n in node.near if n.geom_type == node.geom_type]),
    "horizontal_relatives": horizontal_relatives, # python counter of dict
    "vertical_relatives": vertical_relatives,
}
    return Contextural_features | Cluster_features
def get_cluster_features(cluster):
    cluster_bbox, cluster_cp = get_cluster_distribution(cluster)
    bboxs = np.stack([node.geom_info["bbox"] for node in cluster], axis =1)
    cps = (bboxs[0] + bboxs[1])/2 
    distances_to_cluster_cp = np.linalg.norm(cps - cluster_cp, axis = 1)
    variances_of_distance_to_cluster_cp = np.var(distances_to_cluster_cp)
    # variances of cp coordinate across xyz
    variances_of_cp = np.round(np.var(cps, axis=0), decimals=round_to)
    # Mean height, width and depth of the cluster element
    # variances of orientation
    orientations = np.array([node.principal_axes for node in cluster])
    var_x = np.var(orientations[:,0], axis=0)
    var_y = np.var(orientations[:,1], axis=0)
    var_z = np.var(orientations[:,2], axis=0)
    print(var_x,var_y,var_z)

    cluster_features ={
    "cluster_size": len(cluster),
    "cluster_cp": cluster_cp,
    "cluster_X_start": cluster_bbox[0][0],
    "cluster_X_end": cluster_bbox[1][0],
    "cluster_Y_start": cluster_bbox[0][1],
    "cluster_Y_end": cluster_bbox[1][1],
    "cluster_Z_start": cluster_bbox[0][2],
    "cluster_Z_end": cluster_bbox[1][2],
    "variances_of_cp": variances_of_cp,
    "variances_of_distance_to_cluster_cp": variances_of_distance_to_cluster_cp,
    "mean_height": 0,
    "std_height": 0,
    "orientation_variance": 0,
    "z_alignment_consistency": len([node for node in cluster if node.z_axis_aligned]) / len(cluster),
    }
    return cluster_features
def get_oobb(node):
    vertex = node.geom_info["vertex"]
    _, principal_axes, min_max_bounds = C.oobb_pca(vertex, n_components=3)
    # Rearange the axis to best match world X,Y,Z axis 
    similarity = np.abs(principal_axes)
    row_ind, col_ind = linear_sum_assignment(-similarity)
    min_max_bounds = min_max_bounds[:,col_ind]
    return similarity[col_ind], min_max_bounds[1] - min_max_bounds[0]
def assign_neighbours(node, atol = 0.01):
    """
    1. Get the upper, lower, left, right neighbours of an element based on comparing centre point
    2. Compare the neighbours to yourself if you are upper, lower, 
        techeotically you can not be lefter or righter than your  neighbours return None
    """
    bbox = node.geom_info["bbox"]
    O = GP.get_centre_point(bbox)
    principal_axes = node.principal_axes
    neighbours = node.near + [node]
    # Get the centre points of the neighbours and their bounding box
    cps = np.array([GP.get_centre_point(node.geom_info["bbox"]) for node in neighbours])
    bbox_arrays = np.array([node.geom_info["bbox"] for node in neighbours])
    # Get the upper, lower,by measuring the AABB corners
    upper = neighbours[np.argmax(bbox_arrays[:, 1, 2])]
    lower = neighbours[np.argmin(bbox_arrays[:, 0, 2])]
    # Get the left, right neighbours, in fact, left or right doesnt matter but tne most left and right does so we assign one of them
    scalars = np.linalg.norm(principal_axes, axis=1)
    horizontal_direction = principal_axes[np.argmax(scalars)]
    projection = (cps - O) @ horizontal_direction.T
    left = neighbours[np.argmin(projection)]
    right = neighbours[np.argmax(projection)] 

    # Compare the neighbours to yourself if you are upper, lower, left, right
    if abs(upper.geom_info["bbox"][1][2] - bbox[1][2]) < atol or upper.guid == node.guid:
        upper = None
    elif abs(lower.geom_info["bbox"][0][2] - bbox[0][2]) < atol or lower.guid == node.guid:
        lower = None
    # *** Generally you can not be more left or right than your neighbours ***
    # Exeption is if you are a wall and you encomprises your neighbours aka maybe a window. Here we add a remark
    # in that case, it would return yourself as left or rightneighbour so we just need to test test if the 
    elif left.guid == node.guid:
        left = None
    elif right.guid == node.guid:
        right = None
    results = [n.geom_type if n != None else None for n in [upper, lower, left, right]]
    return results
def get_base_info(node):
    vertex = node.geom_info["vertex"]
    face = node.geom_info["face"]
    lowest_z = node.geom_info["bbox"][0][2]
    base_v_idx =np.where(vertex[:,2] == lowest_z)
    base_f_idx = face[np.all(np.isin(face,base_v_idx), axis =1) == True]
    base_f = vertex[base_f_idx]
    number_of_vertices_in_base = len(base_v_idx[0])
    AABB_base_area = 0
    if number_of_vertices_in_base >=3:
        for f in base_f:
            AABB_base_area += GP.get_polygon_area(f)
    return number_of_vertices_in_base, np.round(AABB_base_area, decimals= round_to)
def is_axis_aligned(node, atol=1e-3, threshold=0.9):
    v = node.geom_info["vertex"]
    f = node.geom_info["face"]
    # Compute face normals
    v1 = v[f[:, 1]] - v[f[:, 0]]
    v2 = v[f[:, 2]] - v[f[:, 0]]
    normals = np.cross(v1, v2)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)  # normalize
    valid = norms[:, 0] > 1e-6 # avoid division by zero
    normals[valid] = normals[valid] / norms[valid]
    bad_faces = np.where(norms[:, 0] < 1e-6)[0]
    # if len(bad_faces) > 0:
    #     print("Degenerate face indices:", bad_faces)
    #     print(node.geom_type)
    #     print(v[f[bad_faces]])
    # Check if normals are close to axis directions
    axis_dirs = np.array([[1,0,0], [0,1,0], [0,0,1],
                          [-1,0,0], [0,-1,0], [0,0,-1]])
    aligned_count = 0
    for n in normals:
        if np.any(np.all(np.abs(n - axis_dirs) < atol, axis=1)):
            aligned_count += 1
    ratio = aligned_count / len(normals)
    return ratio > threshold
def get_horizontal_relatives(graph, node, extent = 0.05):
    """
    1. Reduce the bounding box of the element by a scale factor and use this as collision test
        to see what is the most types it hits the counts and the most common type is the one that is the most likely to be the same type as the element.
    """
    world_max = graph.bbox.copy()
    bbox = node.geom_info["bbox"]
    bbox_z_centre = (bbox[1][2] + bbox[0][2]) / 2
    world_max[0][2] = bbox_z_centre -  extent
    world_max[1][2] = bbox_z_centre +  extent

    bvh_query = graph.bvh_query(world_max)
    bvh_query_types = Counter([graph[guid].geom_type for guid in bvh_query])
    return bvh_query_types
def get_vertical_relatives(graph, node, extent = 0.05):
    world_max = graph.bbox.copy()
    bbox = node.geom_info["bbox"]
    bbox_x_centre = (bbox[1][0] + bbox[0][0]) / 2
    bbox_y_centre = (bbox[1][1] + bbox[0][1]) / 2
    world_max[0][0] = bbox_x_centre - extent
    world_max[1][0] = bbox_x_centre + extent
    world_max[0][1] = bbox_y_centre - extent
    world_max[1][1] = bbox_y_centre + extent
    bvh_query = graph.bvh_query(world_max)
    bvh_query_types = Counter([graph[guid].geom_type for guid in bvh_query])
    return bvh_query_types
def get_cluster(node):
    stack = [node]
    cluster = set()
    while stack:
        current = stack.pop()
        cluster.add(current)
        for n in current.near:
            if n.geom_type == current.geom_type and n not in cluster:
                stack.append(n)
    return cluster
def get_cluster_distribution(cluster):
    bboxs = np.vstack([node.geom_info["bbox"] for node in cluster])
    cps = np.array([GP.get_centre_point(bbox) for bbox in bboxs])
    cluster_bbox = np.array((np.min(bboxs, axis=0), np.max(bboxs, axis=0)))
    cluster_cp = GP.get_centre_point(cluster_bbox)
    return cluster_bbox, cluster_cp
def is_z_axis_aligned(node, atol = 1e-2):
    """
    Computes if the element is tilted or not
    """
    if np.isclose(node.principal_axes[2][2],1, atol=atol):
        return True
    return False