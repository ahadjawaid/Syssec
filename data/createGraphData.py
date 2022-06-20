import os
from dataProcess import *

def createMutagGraphs():
    '''
    returns -> { graphID: [ NodeIDs ] }, where NodeIDs are of the form:
               { nodeID:  ( src_type, edge_type, dst_type ): src_id, dst_id },

               [ graphLabels ]
    
    '''
    nodeLabelPath = os.path.join(os.path.dirname(__file__), 'MUTAG/MUTAG_node_labels.txt')
    graphIndicatorPath = os.path.join(os.path.dirname(__file__), 'MUTAG/MUTAG_graph_indicator.txt')
    graphLabelPath = os.path.join(os.path.dirname(__file__), 'MUTAG/MUTAG_graph_labels.txt')
    edgesPath = os.path.join(os.path.dirname(__file__), 'MUTAG/MUTAG_A.txt')
    edgeLabelPath = os.path.join(os.path.dirname(__file__), 'MUTAG/MUTAG_edge_labels.txt')

    edgesFile = open(edgesPath, 'r')
    edgesLabelFile = open(edgeLabelPath, 'r')
    nodeLabelFile = open(nodeLabelPath, 'r')
    graphIndicatorFile = open(graphIndicatorPath, 'r')
    graphLabelFile = open(graphLabelPath, 'r')

    ntypeDict = createNtypeDict(nodeLabelFile)
    nodeDict = createNodeDict(edgesFile, edgesLabelFile, ntypeDict)
    graphDict = createGraphs(graphIndicatorFile, nodeDict)

    graphLabels = []
    for label in graphLabelFile:
        graphLabels.append(int(label))

    edgesFile.close()
    edgesLabelFile.close()
    nodeLabelFile.close()
    graphLabelFile.close()
    graphIndicatorFile.close()

    return graphDict, graphLabels

def createMUTAGNodes():
    nodeLabelPath = os.path.join(os.path.dirname(__file__), 'MUTAG/MUTAG_node_labels.txt')
    graphIndicatorPath = os.path.join(os.path.dirname(__file__), 'MUTAG/MUTAG_graph_indicator.txt')
    graphLabelPath = os.path.join(os.path.dirname(__file__), 'MUTAG/MUTAG_graph_labels.txt')
    edgesPath = os.path.join(os.path.dirname(__file__), 'MUTAG/MUTAG_A.txt')
    edgeLabelPath = os.path.join(os.path.dirname(__file__), 'MUTAG/MUTAG_edge_labels.txt')

    edgesFile = open(edgesPath, 'r')
    edgesLabelFile = open(edgeLabelPath, 'r')
    nodeLabelFile = open(nodeLabelPath, 'r')
    graphIndicatorFile = open(graphIndicatorPath, 'r')
    graphLabelFile = open(graphLabelPath, 'r')

    ntypeDict = createNtypeDict(nodeLabelFile)
    node_data, node_labels = createNodeData(edgesFile, edgesLabelFile, graphIndicatorFile, graphLabelFile, ntypeDict)


    edgesFile.close()
    edgesLabelFile.close()
    nodeLabelFile.close()
    graphLabelFile.close()
    graphIndicatorFile.close()

    return node_data, node_labels