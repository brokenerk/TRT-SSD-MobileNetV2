#https://www.minds.ai/post/deploying-ssd-mobilenet-v2-on-the-nvidia-jetson-and-nano-platforms
import graphsurgeon as gs
import numpy as np

path = 'model/frozen_inference_graph.pb'
TRTbin = 'TRT_ssd_mobilenet_v2_coco_2018_03_29.bin'
output_name = ['NMS']
dims = [3,300,300]
layout = 7

def add_plugin(graph):
    all_assert_nodes = graph.find_nodes_by_op("Assert")
    graph.remove(all_assert_nodes, remove_exclusive_dependencies=True)

    all_identity_nodes = graph.find_nodes_by_op("Identity")
    graph.forward_inputs(all_identity_nodes)

    Input = gs.create_plugin_node(
        name="Input",
        op="Placeholder",
        shape=[1, 3, 300, 300]
    )

    PriorBox = gs.create_plugin_node(
        name="GridAnchor",
        op="GridAnchor_TRT",
        minSize=0.2,
        maxSize=0.95,
        aspectRatios=[1.0, 2.0, 0.5, 3.0, 0.33],
        variance=[0.1,0.1,0.2,0.2],
        featureMapShapes=[19, 10, 5, 3, 2, 1], # Resolution 300
        #featureMapShapes=[29, 15, 8, 4, 2, 1], # Resolution 450
        numLayers=6
    )

    NMS = gs.create_plugin_node(
        name="NMS",
        op="NMS_TRT",
        shareLocation=1,
        varianceEncodedInTarget=0,
        backgroundLabelId=0,
        confidenceThreshold=1e-8,
        nmsThreshold=0.6,
        topK=100,
        keepTopK=100,
        numClasses=2,
        inputOrder=[0, 2, 1],
        confSigmoid=1,
        isNormalized=1
    )

    concat_priorbox = gs.create_node(
        "concat_priorbox",
        op="ConcatV2",
        axis=2
    )

    concat_box_loc = gs.create_plugin_node(
        "concat_box_loc",
        op="FlattenConcat_TRT",
        axis=1,
        ignoreBatch=0
    )

    concat_box_conf = gs.create_plugin_node(
        "concat_box_conf",
        op="FlattenConcat_TRT",
        axis=1,
        ignoreBatch=0
    )

    namespace_plugin_map = {
        "MultipleGridAnchorGenerator": PriorBox,
        "Postprocessor": NMS,
        "Preprocessor": Input,
        "Cast": Input,
        "image_tensor": Input,
        "Concatenate": concat_priorbox,
        "concat": concat_box_loc,
        "concat_1": concat_box_conf
    }

    for node in graph.find_nodes_by_op('FusedBatchNormV3'):
        gs.update_node(node, op='FusedBatchNorm')

    graph.collapse_namespaces(namespace_plugin_map)
    graph.remove(graph.graph_outputs, remove_exclusive_dependencies=False)
    graph.find_nodes_by_op("NMS_TRT")[0].input.remove("Input")

    # Create a constant Tensor and set it as input for GridAnchor_TRT
    data = np.array([1, 1], dtype=np.float32) 
    anchor_input = gs.create_node("AnchorInput", "Const", value=data)  
    graph.append(anchor_input)
    graph.find_nodes_by_op("GridAnchor_TRT")[0].input.insert(0, "AnchorInput")

    return graph
