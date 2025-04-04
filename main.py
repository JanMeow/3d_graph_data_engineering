import numpy as np
import ifcopenshell
from pathlib import Path
from utils import  Graph
import compute_proxy as CP
# ===================================================================================
# Global Variables for import and export file paths
# ===================================================================================
ifc_folder = Path("data")/"ifc"
ifc_folder.mkdir(parents=True, exist_ok=True)
ifc_path = ifc_folder/"test1.ifc"
export_path = "data/ifc/new_model.ifc"
# ===================================================================================
# ===================================================================================
# ====================================================================
# Main function to run the script
# ====================================================================
def main():
    # Read the IFC file:
    # ====================================================================
    model = ifcopenshell.open(ifc_path)
    root = model.by_type("IfcProject")[0]

    # # Create a graph and establish BVH Tree
    # ====================================================================
    graph = Graph.create(root)
    graph.build_bvh()
    # node = graph.node_dict[guid]
    # queries = graph.bvh_query(node.geom_info["bbox"])

    # Build the graph based on relationship
    # ====================================================================
    for node in graph.node_dict.values():
        if node.geom_info != None:
            node.near = [graph.node_dict[guid] for guid in graph.bvh_query(node.geom_info["bbox"])
                         if guid != node.guid]
    
    guid = "1QmrSv$7f8vB34GVkkI57J"
    node = graph.node_dict[guid]
    intrinsic_features = CP.get_Intrinsic_features(graph, node)
    contextual_features = CP.get_contextural_features(graph, node)
    print("Intrinsic features: ", intrinsic_features)
    print("Contextual features: ", contextual_features)

    return intrinsic_features, contextual_features

if __name__ == "__main__":
    main()