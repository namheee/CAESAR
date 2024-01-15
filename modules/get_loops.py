
"""
Copyright:

This code is originally writeen by {Stack Overflow}
url = {https://stackoverflow.com/questions/31034730/graph-analysis-identify-loop-paths}
"""

# breadth first search of paths and unique loops
def get_loops(adj, paths, maxlen):
    maxlen -= 1
    path_list = []
    for path in paths['paths']:
        for nx_t in adj[path[-1]]:
            nx_path = path + [nx_t]
            if path[0] == nx_t:
                paths['loops'].append(nx_path)
            elif nx_t not in path:
                path_list.append(nx_path)
    paths['paths'] = path_list
    if maxlen == 0:
        return paths
    else:
        return get_loops(adj, paths, maxlen)