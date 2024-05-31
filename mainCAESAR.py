import os
import sys

from collections import defaultdict, Counter
import pickle
import numpy as np
import itertools
import copy
import networkx as nx
import pandas as pd
import time
import shutil
import argparse

from pyboolnet.file_exchange import bnet2primes, primes2bnet
from pyboolnet.digraphs import _primes2signed_digraph
from pyboolnet.prime_implicants import find_constants

from CAESAR.alg_function_opt2 import BLDOI, str2dict, merge_dict, pert_dict2str, pert_str2dict_list, remained_, isnotinfbl, is_init_satisfied
from CAESAR.alg_main_function_opt4 import make_pert_list_new, perturbation_single, returnFBL, reduce_primes 

from multiprocessing import Pool, cpu_count
from contextlib import contextmanager

@contextmanager
def poolcontext(*args, **kwargs):
    pool = Pool(*args, **kwargs)
    yield pool
    pool.terminate()

def merged(model_file, primes, pert_i, pert_str, init_state, pair, target_depc, node_fbl):
    # is_target : Correct or Wrong
    # is_same : is the same when comparing with LDOI
    is_target, is_same = perturbation_single(model_file, primes, pair, pert_i, pert_str, init_state, target_depc, node_fbl)
    return(tuple([pair, pert_str, is_target, is_same]))


def resetting():
    try: shutil.rmtree('./resultLOG/'); os.mkdir('./resultLOG/');
    except FileNotFoundError: os.mkdir('./resultLOG/')

    try: shutil.rmtree('./reduced/'); os.mkdir('./reduced/');
    except FileNotFoundError: os.mkdir('./reduced/')     

    
# check pair list
def define_pairList(find_allcombi, st_pair_idx, meta_idx, meta_idx_reverse):
    if find_allcombi == 'all':
         check_pairList = st_pair_idx
    else:
        check_pairList = []
        finding_opt = input('att? or att_idx?:')
        if finding_opt == 'att':
            s_att = input('s_att:')
            s_idx = meta_idx[tuple([s_att])]        
            t_att = input('t_att:')
            t_idx = meta_idx[tuple([t_att])]     
            print(s_idx,t_idx)
            check_pairList.append(tuple([s_idx,t_idx]))
     
        else: 
            s_idx = input('init_idx:')
            t_idx = input('target_idx:')
            try: s_idx = int(s_idx); t_idx = int(t_idx)        
            except ValueError: pass                 
            check_pairList.append(tuple([s_idx,t_idx]))
    return check_pairList


def main(start, find_all, networkName, primes, paired_info, result_fname):    
    desired_pstate, ctrl_cand, meta_idx_reverse, node_fbl_info = paired_info
    model_file = './network/' +networkName+'.bnet' 
    logf = open('./Result/'+'log_'+str(networkName)+'.txt','w')
    # ============================================================================
    # main
    # ============================================================================
    paired_result = pd.DataFrame([])        
    for pair, target_depc in desired_pstate.items():
        resetting()
        
        if find_all == 'DEPC':
            pert_list_all = make_pert_list_new(primes, ctrl_cand[pair])
        elif find_all == 'all':
            all_ = list(primes.keys())
            f_all = {k:ctrl_cand[pair][k] for k in all_}
            pert_list_all = make_pert_list_new(primes, f_all)
            #print(pert_list_all)
        else: # for checking individual node
            print('node:',find_all)
            find_all = find_all.split(',')
            f_all = {k:ctrl_cand[pair][k] for k in find_all}
            pert_list_all = make_pert_list_new(primes, f_all)
  
        init_state = meta_idx_reverse[pair[0]][0]
        node_fbl = node_fbl_info[pair]

        p_num = 1
        while 1:
            print('p_num:',p_num)
            if p_num == 5: break #max target number
            pert_list = [x for x in pert_list_all if len(x) == p_num]    
            pert_list_str = [pert_dict2str(m) for m in pert_list]              
            
            arg0 = [model_file for _ in range(len(pert_list))]
            arg1 = [primes for _ in range(len(pert_list))]            
            arg2 = [pert_i for pert_i in pert_list]
            arg3 = [pert_list_str[idx] for idx in range(len(pert_list))]
            arg4 = [init_state for _ in range(len(pert_list))]
            arg5 = [pair for _ in range(len(pert_list))]
            arg6 = [target_depc for _ in range(len(pert_list))]
            arg7 = [node_fbl for _ in range(len(pert_list))]          

            num_cores = 10
            with poolcontext(processes = num_cores) as pool:
                result = pool.starmap(merged, zip(arg0,arg1,arg2,arg3,arg4,arg5,arg6,arg7))        
            if np.any('Correct'==pd.DataFrame(result).iloc[:,2]):
                paired_result = pd.concat([paired_result,pd.DataFrame(result).loc[('Correct'==pd.DataFrame(result).iloc[:,2]).values,:]])
                runtime = round((time.time() - start),3)
                print("***run time(sec) :",runtime)
                print('nodeNum_'+str(p_num)+'_'+str(runtime), file = logf)
                logf.close()
                break
            else:
                p_num += 1
                continue
            
    # ============================================================================
    # summary
    # ============================================================================
    if paired_result.shape[0] > 0 :
        paired_result.columns = ['pair','target','ans','ldoi']    
        
        pert_result_all = paired_result.groupby('pair')['target'].apply(list)
        compare_result_all = paired_result.groupby('pair')['ldoi'].apply(list)

        results = []
        for key_name in (paired_result.pair):
            r1 = tuple(pert_result_all[key_name])
            
            compare_ = tuple(compare_result_all[key_name]) 
            desired_ = tuple(desired_pstate[key_name].items())

            results.append([r1, compare_, desired_])


        results_pd = pd.DataFrame(results, index=list(paired_result.pair), columns = ['Alg','compare_ldoi','desired_DEPCstate'])
        results_pd = results_pd.drop_duplicates()
        results_pd.to_csv(result_fname)
        #print(networkName)
        #print(results_pd)
        return list(results_pd.Alg),runtime 
        #logf.close()
    else:
        print('No answer')
        return [], 0 

def caesar(networkName, source, target, D_state=None, pair=(0,1), fbl_thres=5):
    
    # ====================
    # default parameter
    # ====================
    fbl_thres = fbl_thres 
    print('fbl_threshold:',fbl_thres)
    find_allcombi = 'all'
    find_all = 'DEPC' # 'all'


    # ========
    # setting
    # ========
    # resetting()
    if 'Simulation_updated' not in os.listdir(): os.mkdir('./Simulation_updated/')    
    if 'Result' not in os.listdir(): os.mkdir('./Result/')        
    
    networkName = [x for x in [networkName] if x+'.bnet' in os.listdir('./network')][0]
    print('-> networks:',networkName)

    
    start = time.time()
    model_file = './network/' +networkName+'.bnet'        
    primes = bnet2primes(model_file)
    graph = _primes2signed_digraph(primes)
    nodeList = list(primes.keys())
    result_fname = './Result/'+networkName+'_Result_' +str(bool(find_all))+'.csv'

    st_pair_idx = [(0,1)]
    meta_idx = {tuple([source]):0,tuple([target]):1}
    meta_idx_reverse = {0:tuple([source]),1:tuple([target])}
    
    check_pairList = define_pairList(find_allcombi, st_pair_idx, meta_idx, meta_idx_reverse)
    print(check_pairList)

    # ============================================================================
    def searchFBL(check_pairList, fbl_thres):
        
        desired_pstate = defaultdict(dict) # desired phenotypc state (=DEPC)
        ctrl_cand = defaultdict(dict) # control_candidate
        node_fbl_info = defaultdict(dict) # all fbls
        
        for pair in check_pairList:
            s_att = meta_idx_reverse[pair[0]]
            t_att = meta_idx_reverse[pair[1]]
            s_att_np = np.array([int(x) for x in s_att[0]])
            t_att_np = np.array([int(x) for x in t_att[0]])

            # FBL-1: DEPC
            candidates = [x for x,y in zip(nodeList,(s_att_np-t_att_np != 0)) if y]
            subg = graph.subgraph(candidates)
            largest_sub = max(nx.strongly_connected_components(subg), key=len)
            maxlen = len(largest_sub)
            depc_fbl = returnFBL(subg, maxlen) 
            subnode_fbl = depc_fbl.copy()

            # FBL-2
            subnode_fbl += returnFBL(graph, fbl_thres)        
            #subnode_fbl = returnFBL(graph.subgraph(largest), 3)        

            # FBL-3: ~DEPC
            try:        
                subg = graph.subgraph([x for x,y in zip(nodeList,(s_att_np-t_att_np == 0)) if y])
                largest_sub = max(nx.strongly_connected_components(subg), key=len)
                maxlen = len(largest_sub)
                subnode_fbl += returnFBL(subg, maxlen)
            except ValueError:
                pass

            node_fbl_info[pair] = list(set(subnode_fbl)) 

            # updated ##########################
            ctrl_cand[pair] = {x:y for x,y in zip(nodeList, t_att_np)}
                        
            for fbl in depc_fbl:
                target_state = {x:y for x,y in zip(fbl, t_att_np[[nodeList.index(x) for x in fbl]])}  
                desired_pstate[pair].update(target_state)  

        return desired_pstate, ctrl_cand, node_fbl_info

    desired_pstate, ctrl_cand, node_fbl_info = searchFBL(check_pairList, fbl_thres)
    
    # updated ##########################
    if D_state != None:
        desired_pstate[pair] = (D_state)
        print('User-defined D_state:',D_state)
        
    # save for validation ===========================================================================
    netFilename_new = './Simulation_updated/'+networkName+ 'Simulation.pickle'
    result_new = (primes, node_fbl_info, meta_idx_reverse, desired_pstate, ctrl_cand)
    with open(netFilename_new,'wb') as f: pickle.dump(result_new, f)

    # run =============================================================================================
    paired_info = (desired_pstate, ctrl_cand, meta_idx_reverse, node_fbl_info)
    alg, runtime = main(start, find_all, networkName, primes, paired_info, result_fname)
    print('Answer:',alg)
    return desired_pstate, alg, runtime, node_fbl_info
    #return desired_pstate, alg, runtime