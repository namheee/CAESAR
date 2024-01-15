# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 13:39:10 2022

@author: NamheeKim
"""

from collections import defaultdict, Counter
import numpy as np
import itertools
import networkx as nx
from modules.expanded_network import PlayWithExpNet, to_sif, form_network, Get_expanded_network, update_single_DOI

#%%

def str2dict(initstate_str, primes):
    nodeList = list(primes.keys())
    return {x:int(y) for x,y in zip(nodeList, initstate_str)}


def merge_dict(ind_dic_list):
    merged_dict = dict()
    for _, di in enumerate(ind_dic_list):
        if di.items() not in merged_dict.items():
            merged_dict.update(di)
    return merged_dict


def pert_dict2str(pert_i, annotType = 'int'):
    if annotType == 'str':      
        return ('&'.join(sorted(['~'+x if y == 'false' else x for x,y in pert_i.items()])))
    else:
        return ('&'.join(sorted(['~'+x if y == 0 else x for x,y in pert_i.items()])))


def pert_str2dict(pert_str):
    if '~' in pert_str: 
        return({pert_str[1:]:0})
    else:
        return({pert_str:1})


def pert_str2dict_list(pert_i):
    pert_dict = []   
    for x in pert_i:
        if '&' in x:
            tmp_dict = merge_dict([pert_str2dict(y) for y in x.split('&')])
            if tmp_dict not in pert_dict:
                pert_dict.append(tmp_dict)
        else:
            tmp_dict = pert_str2dict(x)
            if tmp_dict not in pert_dict:
                pert_dict.append(tmp_dict)  
    return pert_dict
    
    
def is_ssc(cycle_list):
    # is self-stabilizing coponent
    cnodes = [x for x in cycle_list if '&' in x] # composite node
    if len(cnodes) == 0:
        return True
    else:
        ssc = []
        for cnode in cnodes:     
            if (set(cnode.split('&'))-set(cycle_list) == set()):
                ssc.append(True)
            else:
                ssc.append(False)
        return np.all(ssc)

def BLDOI(networkname):
    # from expanded network & change node names
    # using BooleanDOI
    pen = PlayWithExpNet(networkname)
    enet = pen.get_expanded_network()
    
    # compute LDOI =============================================
    singleDOI_filename = networkname.split('.txt')[0]+'0'
    prefix,suffix='n','n'
    f = open(networkname,'r')
    lines = f.readlines()
    f.close()
    Gread, readnodes = form_network(lines, sorted_nodename = False)
    node_mapping = {prefix+str(index)+suffix:node_name for index, node_name in zip([i for i in range(len(readnodes))], readnodes)}
    node_mapping.update({'~'+prefix+str(index)+suffix:'~'+node_name for index, node_name in zip([i for i in range(len(readnodes))], readnodes)})

    # form expanded network
    G_expanded = Get_expanded_network(Gread, prefix=prefix, suffix=suffix)

    # initialization
    TDOI_BFS, flagged, potential = {}, {}, {} #Domain of Influence

    # first step, search through DOI of single node
    update_single_DOI(G_expanded, singleDOI_filename, TDOI_BFS, flagged, potential)
    
    TDOI_Dict = defaultdict(list)
    f = open(singleDOI_filename+'_TDOI.txt','r')
    for l in f:
        line = l.split('\n')[0]
        str2dict = {node_mapping[line.split(' : ')[0]]: [node_mapping[x] for x in set([line.split(' : ')[0]] + line.split(' : ')[1].split("'")) if (('set()' not in x) &(', ' not in x) &('_' not in x) & ('{' not in x) & ('}' not in x) )]}
        TDOI_Dict.update(str2dict)
    f.close()
    #  ==========================================================
    return enet, TDOI_Dict
    

# condition
def isnotinfbl(fbl1, fbl):
    isin = []
    for fx in fbl:
        if set(fx) != set(fbl1):
            isin.append(len(set(fbl1) - set(fx)))
    if 0 in isin: return False
    else: return True


    

def propagate_inG(G, fixed_ldoi_reduced, init_fixed):
    prop_region = set(itertools.chain(*[fixed_ldoi_reduced[n] for n in init_fixed])) & set(G.nodes)     
    prop_node = set([x for x in G.nodes if set(x.split('&')) - prop_region == set()])
    while 1:
        prop_n = prop_node.copy()
        for node in prop_node:
            prop_n = prop_n | set(G.nodes)&set(itertools.chain(*[fixed_ldoi_reduced[bfs_edge[1]] for bfs_edge in nx.bfs_edges(G, source=node, reverse = False, depth_limit = 1)]))       
        prop_n = prop_n | set([x for x in G.nodes if set(x.split('&')) - prop_n == set()])
        if len(prop_node) != len(prop_n):
            prop_node = prop_n
            continue
        else: break
    return prop_n


def is_init_satisfied(enet, node, fixed_ldoi_reduced, init_str_p):
    sub_n = nx.algorithms.dag.ancestors(enet, node) | {node}      
    enet_sub = enet.subgraph(sub_n)
    init_satisfied = node in propagate_inG(enet_sub, fixed_ldoi_reduced, init_str_p)
    return init_satisfied
   

     
def remained_(reduced_bnet, reduced_num, node_fbl, pert_str, init_str_p):
    networkname = './reduced//reduced_bnet_'+pert_str+'_'+str(reduced_num)+'.txt'
    with open(networkname, 'w') as f: #bnet to 
        reduced_bnet = reduced_bnet.replace(',','*=')
        reduced_bnet = reduced_bnet.replace('!','not ')
        reduced_bnet = reduced_bnet.replace('|','or')
        reduced_bnet = reduced_bnet.replace('&','and')
        reduced_bnet = reduced_bnet.replace('\n\n','\n')
        f.write(reduced_bnet)
 
    try:
        enet, TDOI_Dict = BLDOI(networkname)
        # to_sif (write network structure file)
        # to_sif(enet, file_prefix='./reduced/reduced_bnet_'+pert_str+'.bnet')

        remained_fbl_dict = defaultdict(dict)
        remained_fbl_dict_info = defaultdict(dict)
        remained_fbl = [x for x in node_fbl if set(x)-set(enet.nodes) == set()] 
      
        for fbl1 in remained_fbl:
            remained_fbl_dict[fbl1] = defaultdict(list)
            remained_fbl_dict_info[fbl1] = defaultdict(list)            
            
            subnodes = [x for x in enet.nodes if set([y[1:] if '~' in y else y for y in x.split('&')])&set(fbl1) != set()]    
            enet_sub_cycle = enet.subgraph(subnodes)    
            
            ssc_cycle_stabilized = False
            is_oscillating = False
            for cycle1 in nx.simple_cycles(enet_sub_cycle):    
                if len(set(fbl1) - set([n[1:] if ('~' in n) else n for n in set(itertools.chain(*[n.split('&') for n in cycle1]))])) != 0:
                    continue
                cycle1 = tuple(max(nx.strongly_connected_components(enet.subgraph(set(itertools.chain(*[x.split('&') for x in cycle1])) | set(cycle1))),key=len)) # to include maximal one
                composite = [x for x in cycle1 if '&' in x]              
                non_composite = [x for x in cycle1 if '&' not in x] 
                if is_ssc(cycle1):
                    remained_fbl_dict_info[fbl1]['ssc'].append(tuple(sorted(cycle1)))

                    if len(composite) != 0:     
                        if np.all([np.all([is_init_satisfied(enet, c2, TDOI_Dict, init_str_p) for c2 in c1.split('&')]) for c1 in composite]):
                            ssc_cycle_stabilized = True
                            remained_fbl_dict[fbl1]['ssc'].append(tuple(sorted(cycle1)))           
                        else: continue
                    else: 
                        if is_init_satisfied(enet, non_composite[0], TDOI_Dict, init_str_p):
                            ssc_cycle_stabilized = True
                            remained_fbl_dict[fbl1]['ssc'].append(tuple(sorted(cycle1)))
                        else: continue
                    
                elif (2 in list(Counter([n[1:] if '~' in n else n for n in set(itertools.chain(*[n.split('&') for n in cycle1]))]).values())): 
                    is_oscillating = True
                    continue  
                
                else: #csm
                    remained_fbl_dict_info[fbl1]['csm'].append(tuple(sorted(cycle1)))
                    composite = [x for x in cycle1 if '&' in x]              
                    if np.all([np.all([is_init_satisfied(enet, c2, TDOI_Dict, init_str_p) for c2 in c1.split('&')]) for c1 in composite]):
                        remained_fbl_dict[fbl1]['csm'].append(tuple(sorted(cycle1)))                    
                    else:
                        continue            

            remained_fbl_dict[fbl1]['ssc'] = list(set(remained_fbl_dict[fbl1]['ssc']))
            remained_fbl_dict[fbl1]['csm'] = list(set(remained_fbl_dict[fbl1]['csm']))
            remained_fbl_dict_info[fbl1]['ssc'] = list(set(remained_fbl_dict_info[fbl1]['ssc']))
            remained_fbl_dict_info[fbl1]['csm'] = list(set(remained_fbl_dict_info[fbl1]['csm']))               
            remained_fbl_dict[fbl1]['is_oscillating'].append(is_oscillating)
            remained_fbl_dict[fbl1]['ssc_stabilized'].append(ssc_cycle_stabilized)
            
        remained_fbl_dict_info = {x:y for x,y in remained_fbl_dict_info.items() if np.any([set(y['ssc']) != set(), set(y['csm']) != set()])}
        remained_fbl_dict = {x:y for x,y in remained_fbl_dict.items() if np.any([set(y['ssc']) != set(), set(y['csm']) != set()])}
        return networkname, remained_fbl_dict, remained_fbl_dict_info, False, TDOI_Dict
    
    except KeyError:
        print('Error')
        error = True
        return './reduced/reduced_bnet_'+pert_str+'.bnet', {}, error, TDOI_Dict