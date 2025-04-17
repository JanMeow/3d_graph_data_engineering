import numpy as np
import ifcopenshell
from pathlib import Path
from utils import  Graph
# ===================================================================================
# Global Variables for import and export file paths
# ===================================================================================
ifc_folder = Path("data")/"ifc"
ifc_folder.mkdir(parents=True, exist_ok=True)
# ===================================================================================
# Global Variables for features type and rounding
# ===================================================================================
label_map = {
    "None": 0,
    "IfcWall": 1, #Can further classfy into outer or inner
    "IfcSlab": 2,
    "IfcRoof": 3,
    "IfcColumn": 4,
    "IfcBeam": 5,
    "IfcCurtainWall": 6,
    "IfcFooting": 7,
    "IfcPlate": 8
}

round_to = 2
# ===================================================================================
# ===================================================================================
# ====================================================================
# Extracting and cleaniing the IFC file for use
# ====================================================================
def extract_ifc_type_for_training(file, target_type = label_map):
    # Read the IFC file:
    model = ifcopenshell.open(file)
    root = model.by_type("IfcProject")[0]
    # # Create a graph and establish BVH Tree:
    graph = Graph.create(root, target_type=target_type)
    graph.build_bvh()
    # Build the graph based on relationship
    for node in graph.node_dict.values():
        if node.geom_info != None:
            node.near = [graph[guid] for guid in graph.bvh_query(node.geom_info["bbox"])
                         if guid != node.guid]
    return graph
    