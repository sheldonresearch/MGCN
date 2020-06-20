#!usr/bin/env python
# -*- coding:utf-8 _*-
import pandas as pd
import pickle
import community
from collections import defaultdict

def partition(graph, resolution=100000):
    part = community.best_partition(graph, resolution=resolution)
    community_number = max(part.values()) + 1
    print("community number: ", community_number)
    nodes_part_list = []
    com2nodes = defaultdict(list)
    for node, com in part.items():
        com2nodes[com].append(node)

    for com, node_list in com2nodes.items():
        if len(node_list) < 1000:  # 500
            # print('community {} size {} and we ignore it'.format(com, len(node_list)))
            continue
        else:
            print('community {} size {}'.format(com, len(node_list)))
            nodes_part_list.append(node_list)

    print('we have {} communities and each of them is larger than 1000'.format(len(nodes_part_list)))

    # pickle.dump(nodes_part_list, open('./{}.nodes_part_list'.format(network_alias), 'wb'))
    return nodes_part_list




for network_name in ['facebook']:
    """
    network_1 shared_number:1056, 0:1055(含)是公共节点集合, 合并后变成了67个社区
    network_2 shared_number:1138, 0:1137(含)是公共节点集合, 合并后变成了87个社区
    """
    all_parts_name2index = defaultdict(dict)
    global_name2index = defaultdict(int)
    if network_name == 'facebook':
        network_alias = 'network_1'
        shared_number = 1056
    elif network_name == 'twitter':
        network_alias = 'network_2'
        shared_number = 1138

    network_pd = pd.read_csv('../{}_network.csv'.format(network_name))

    """nodes_part_list: a dictionary
    {
        community_id:[node_id,node_id]
    }
    """
    nodes_part_list = pickle.load(open('./{}.nodes_part_list'.format(network_alias), 'rb'))

    g_count = 0
    for part_name, part in enumerate(nodes_part_list):
        name2index_part = defaultdict(int)
        for index, node in enumerate(part):
            name2index_part[node] = index
            if part_name == 0:
                global_name2index[node] = g_count
                g_count = g_count + 1
            elif (part_name > 0) and (index >= shared_number):
                global_name2index[node] = g_count
                g_count = g_count + 1

        pickle.dump(name2index_part, open('./{}_{}.name2index'.format(network_alias, part_name), 'wb'))

        all_parts_name2index[part_name] = name2index_part

    pickle.dump(all_parts_name2index, open('./{}_all_parts.name2index'.format(network_alias), 'wb'))
    pickle.dump(global_name2index, open('./{}_global.name2index'.format(network_alias), 'wb'))



