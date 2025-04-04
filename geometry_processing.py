import numpy as np

"""
This module contains the boolean opserations that are used for merging collision shapes
or touching shapes
# Task 
# Install conda, meshplot,libigl,PolyFem, 

1. Architect inputs IFC good/bad its not production ready anyways
2. Create product classes for the production companies 
3. Corner type menu, 
4. Component Matching Semi Touching the same type on the same floor, knowing floor thickness
5. (Here we are also not asking what if the type is not declared) minimal inner core, edge cases 
like range betweeen 100-150 then persay 300 wall can never be reached


Task for the week:
1. Get all touching wall type in a group
2. Perform boolean operation using the library could be trimesh/ Pymesh

"""
# ====================================================================
# Helpers Functons
# ====================================================================
def align_to_axis(vertex, axis =2):
  if len(np.unique(vertex[:,axis])) == 2:
    return True
  return False
def get_centre_point(bbox):
  return (bbox[0] + bbox[1]) / 2
def is_xzy_box(vertex):
  for i in range(3):
    if len(np.unique(vertex[:,i])) != 2:
        return False
  return True
def get_bbox_dim(bbox):
  return bbox[1] - bbox[0]
def get_base_curve(node):
  vertex = node.geom_info["vertex"]
  lowest_z = node.geom_info["bbox"][0][2]
  return vertex[vertex[:,2] == lowest_z]
def decompose_2D(node):
  base = get_base_curve(node)
  vs = np.array([base[1]- base[0],base[2] - base[1]])
  return vs/ np.linalg.norm(vs, axis = 1)[:, np.newaxis]
def decompose_2D_from_base(base):
  vs = np.array([base[1]- base[0],base[2] - base[1]])
  scalars = np.linalg.norm(vs, axis = 1)
  sort_indices = np.argsort(np.linalg.norm(vs, axis = 1))
  return vs[sort_indices]
def get_unit_vector(v):
  return v/np.linalg.norm(v)
def get_normal(faces):
    v0, v1, v2 = faces[:,0], faces[:,1], faces[:,2]  
    n = np.cross(v1 - v0, v2 - v0)
    return n/np.linalg.norm(n)
def angle_between(v1, v2):
  v1_u = v1/np.linalg.norm(v1)
  v2_u = v2/np.linalg.norm(v2)
  return np.arccos(np.clip(np.dot(v1_u, v2_u.T), -1.0, 1.0))
def project_points_on_face(points, face_normal, face):
   v = points - face[0]
   proj = np.dot(v, face_normal)[:, np.newaxis] * face_normal
   return points - proj
def get_local_coors(T_matrix, vertex):
    inverse = np.linalg.inv(T_matrix.T)
    ones = np.ones(shape = (vertex.shape[0],1))
    result = np.around(np.hstack((vertex, ones)) @ inverse, 2)
    return result[:,0:-1]
def np_intersect_rows(arr1, arr2):
  set0 = set(map(tuple, arr1))
  set1 = set(map(tuple, arr2))
  shared = set0.intersection(set1)
  return np.array(list(shared))
def np_intersect_rows_atol(arr1,arr2, atol = 0.01):
  diffs = np.linalg.norm(arr1[:, None, :] - arr2[None, :, :], axis=2)
  matches = diffs < atol
  # arr1 row i matches arr2 row j
  i,j = np.where(matches)
  return i,j
def get_polygon_area(arr):
  x = arr[:,0]
  y = arr[:,1]
  return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))