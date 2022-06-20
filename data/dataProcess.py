import torch

def createNtypeDict(nodeLabelFile):
    ntypeKeys = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4: 'I', 5: 'Cl', 6: 'Br'}
    ntypeDict = {}
    for nodeID, ntype in enumerate(nodeLabelFile, 1):
        ntypeDict[nodeID] = ntypeKeys[int(ntype)]

    return ntypeDict

def createNodeDict(edgesFile, edgesLabelFile, ntypeDict):
    '''
    returns -> { nodeID:  ( src_type, edge_type, dst_type ): src_id, dst_id }

    Edge_Labels
    -----------
    0  aromatic
    1  single
    2  double
    3  triple
    '''
    edgeKeys = {0: 'aromatic', 1: 'single', 2: 'double', 3: 'triple'}
    nodeDict = {}
    for edge, edge_label in zip(edgesFile, edgesLabelFile):
        edge = tuple(map(int, edge.strip().split(', ')))
        srcNode = edge[0]
        key = (1,  edgeKeys[int(edge_label)], 1)
        srcKey = (1, ntypeDict[edge[0]], 1)
        destKey = (1, ntypeDict[edge[1]], 1)
        edge = torch.tensor(edge)

        if srcNode in nodeDict:
            if key in nodeDict[srcNode]:
                nodeDict[srcNode][key].append(edge)
            elif not(key in nodeDict[srcNode]):
                nodeDict[srcNode][key] = [edge]
            elif srcKey in nodeDict[srcNode]:
                nodeDict[srcNode][srcKey].append(edge)
            elif not(srcKey in nodeDict[srcNode]):
                nodeDict[srcNode][srcKey] = [edge]
            elif destKey in nodeDict[srcNode]:
                nodeDict[srcNode][destKey].append(edge)
            else:
                nodeDict[srcNode][destKey] = [edge]
        else: 
            nodeDict[srcNode] = {key: [edge], srcKey: [edge], destKey: [edge]}
    
    return nodeDict

def createGraphs(graphIndicatorFile, nodeDict):
    '''
    returns -> { graphID: [ NodeIDs ] }, where NodeIDs are of the form:
               { nodeID:  ( src_type, edge_type, dst_type ): src_id, dst_id }
    '''
    graphDict = {}
    for nodeID, graphID in enumerate(graphIndicatorFile, 1):
        graphID = int(graphID)
        if graphID in graphDict:
            node = nodeDict[nodeID]
            for edge in node:
                if edge in graphDict[graphID]:
                    graphDict[graphID][edge].extend(node[edge])
                else:
                    graphDict[graphID][edge] = node[edge]
        else: 
            graphDict[graphID] = nodeDict[nodeID]
    
    return graphDict

def createNodeData(edgesFile, edgesLabelFile, graphIndicatorFile, graphLabelFile, ntypeDict):
    edgeKeys = {0: 'aromatic', 1: 'single', 2: 'double', 3: 'triple'}
    node_data = {}
    for edge, edge_label in zip(edgesFile, edgesLabelFile):
        edge = tuple(map(int, edge.strip().split(', ')))
        key = (ntypeDict[edge[0]],  edgeKeys[int(edge_label)], ntypeDict[edge[1]])
        edge = torch.tensor(edge)

        if key in node_data:
            node_data[key].append(edge)
        else:
            node_data[key] = [edge]

    graph_labels = []
    for label in graphLabelFile:
        graph_labels.append(int(label))

    node_labels = []
    for graphID in graphIndicatorFile:
        node_labels.append(graph_labels[int(graphID) - 1])

    return node_data, node_labels



