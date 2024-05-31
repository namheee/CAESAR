# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 15:57:42 2022

@author: Namhee Kim
"""

from collections import defaultdict, Counter
import numpy as np
import itertools
import copy
import networkx as nx
import pandas as pd
import sys
import igraph as ig
import time
from pyboolnet.file_exchange import bnet2primes, primes2bnet
from pyboolnet.prime_implicants import find_constants, create_constants, percolate

import os
from .modules.expanded_network import PlayWithExpNet, form_network
from .modules.get_loops import get_loops
from .alg_function_opt2 import str2dict, merge_dict, pert_dict2str, pert_str2dict_list, remained_, isnotinfbl, propagate_inG, is_ssm, is_init_satisfied

#%%
# Function for reducing primes
def reduce_primes(primes: dict, constants: dict):
    reduced_primes = create_constants(primes, constants, copy=True)
    logic_reduced_primes = percolate(primes=reduced_primes, remove_constants=False, copy=True, max_iterations= len(primes)*10)
    fix_dict_reduced = find_constants(logic_reduced_primes)
    
    logic_reduced_primes = percolate(primes=reduced_primes, remove_constants=True, copy=True, max_iterations= len(primes)*10)  
    if logic_reduced_primes == {}: logic_reduced = ''
    else: logic_reduced = sorted([x for x in primes2bnet(logic_reduced_primes).split('\n') if x != ''])
    
    return logic_reduced_primes, fix_dict_reduced, logic_reduced


#%%

def is_contradict(state):
    return (2 in list(Counter([n[1:] if '~' in n else n for n in state]).values()))
    
def is_same(fbl1, fbl2):
    not_empty = (len(fbl1) != 0) & (len(fbl2) != 0)
    is_diff_fbl = (set(sorted([x for x in fbl1 if '&' not in x])) - set(sorted([x for x in fbl2 if '&' not in x])) != set()) & (set(sorted([x for x in fbl2 if '&' not in x])) - set(sorted([x for x in fbl1 if '&' not in x])) != set())
    return (not is_diff_fbl)&not_empty
    
def fbl_type(slist):
    Counter_dict = Counter(slist)
    if (Counter_dict[-1])%2 == 0 : return ('pos')
    else: return ('neg')

def make_pert_list_new(primes, target_depc):
    input1 = [{k:v} for k,v in target_depc.items()]
    target_max = 5
    output1 = sum([list(map(list, itertools.combinations(input1, i))) for i in range(target_max + 1)], [])[1:]
    pert_list =  [merge_dict(pc) for pc in output1]    
    return pert_list

#%%
def returnFBL(graph, maxlen):
    fbl_dict = defaultdict(str)
    G = ig.Graph.from_networkx(graph)
    adj = [[n.index for n in v.neighbors(mode='out')] for v in G.vs]
    loops = []
    for start in range(G.vcount()):
        loops += get_loops(adj,  {'paths': [[start]], 'loops': []}, maxlen)['loops']
    for fblist in [[G.vs[int(vs_idx)]['_nx_name'] for vs_idx in loop] for loop in loops]:
        slist = []
        for idx in range(len(fblist)-1):
            slist += list(graph.adj[fblist[idx]][fblist[idx+1]]['sign'])
        fbl_dict[tuple(sorted(fblist[:-1]))] = fbl_type(slist)
    node_fbl = [tuple(sorted(x)) for x,y in fbl_dict.items() if (len(x) <= maxlen) & (y == 'pos')]
    #print('# of node_fbl:',len(node_fbl))
    return node_fbl


#%%

def check_hierarchy_btw_components(primes, exp_G, pert_i, fixed_canal_reduced, compo1_fbl, compo2_fbl, init_dict):
    compo1 = set(itertools.chain(*[x.split('&') for x in compo1_fbl]))
    compo2 = set(itertools.chain(*[x.split('&') for x in compo2_fbl]))
    compo1_composite = set(itertools.chain(*[x.split('&') for x in compo1_fbl if '&'  in x]))
    compo2_composite = set(itertools.chain(*[x.split('&') for x in compo2_fbl if '&'  in x]))
    compo1_no_cond = compo1-(compo1_composite-set(compo1_fbl))
    compo2_no_cond = compo2-(compo2_composite-set(compo2_fbl))
    
    # some parts are in the canalized region after fixing the component =============================================
    compo1_dict = merge_dict(pert_str2dict_list(compo1))
    compo1_dict.update(pert_i)
    _, compo1_dict_wcanal, _ = reduce_primes(primes, compo1_dict)  # LDOI
    
    compo2_dict = merge_dict(pert_str2dict_list(compo2))   
    compo2_dict.update(pert_i)
    _, compo2_dict_wcanal, _ = reduce_primes(primes, compo2_dict)
    
    # other parts are stabilized by initial condition ===============================================================   
    compo1_dict_wcanal_wi = compo1_dict_wcanal.copy()
    compo2_dict_wcanal_wi = compo2_dict_wcanal.copy()
    
    # some nodes can be fixed by canalizing despite its contradiction ===============================================
    sub1 = set([pert_dict2str({x:y}) for x,y in compo1_dict_wcanal_wi.items()]) | compo1
    sub1 = sub1 | set([x for x in exp_G.nodes if (set(x.split('&'))-sub1)==set()]) # => LDOI
    sub1 = sub1 | propagate_inG(exp_G, fixed_canal_reduced, sub1)
    sub1 = sub1 | compo2
    # and then check the path between components
    exp_sG1 = exp_G.copy().subgraph(sub1)
    
    case1_reachable = np.any([nx.has_path(exp_sG1,x,y) for x,y in itertools.product(compo1_no_cond, compo2_no_cond-compo1_no_cond) if x != y]) #'compo1->compo2'        
    case1_compo = np.all([np.any([nx.has_path(exp_sG1,x,y) for x in (set(exp_sG1.nodes) & (compo1 | compo2)) if x!= y]) for y in compo2_composite])
    case1_compo_r = np.all([np.any([nx.has_path(exp_sG1,x,y) for x in compo1_no_cond if x!= y]) for y in compo2_composite & (compo2_no_cond-compo1_no_cond)])

    case1 = case1_reachable & case1_compo & case1_compo_r
    
    sub2 = set([pert_dict2str({x:y}) for x,y in compo2_dict_wcanal_wi.items()]) | compo2
    sub2 = sub2 | set([x for x in exp_G.nodes if (set(x.split('&'))-sub2)==set()]) 
    sub2 = sub2 | propagate_inG(exp_G, fixed_canal_reduced, sub2)
    sub2 = sub2 | compo1
    exp_sG2 = exp_G.copy().subgraph(sub2)
    
    case2_reachable = np.any([nx.has_path(exp_sG2,x,y) for x,y in itertools.product(compo2_no_cond, compo1_no_cond-compo2_no_cond)]) #'compo1->compo2'        
    case2_compo = np.all([np.any([nx.has_path(exp_sG2,x,y) for x in set(exp_sG2.nodes) & (compo1 | compo2) if x!= y]) for y in compo1_composite])
    case2_compo_r = np.all([np.any([nx.has_path(exp_sG2,x,y) for x in compo2_no_cond if x!= y]) for y in compo1_composite & (compo1_no_cond-compo2_no_cond)])

    case2 = case2_reachable & case2_compo & case2_compo_r
    
    
    # ===============================================================
    if (case1 == False)&(case2 == True): compo = compo1
    elif (case1 == True)&(case2 == False): compo = compo2                                                                      
    else: compo = set(compo1 | compo2)    
                    

    return case1, case2, compo, is_same(compo1_fbl, compo2_fbl)

#%%

def is_in_canalizedR(fixed_bycanal, fixed_desired, target_depc, logf):
    if fixed_bycanal!=fixed_desired:
        print(fixed_bycanal, file = logf)
        print(fixed_desired, file = logf)
        print('# ===== Final Result (Type1.1) ===== #', file = logf)
        print('Wrong, some desired nodes are not in the canalized region (contradict).', file = logf)
        logf.close()
        return 'type1'
    elif len(fixed_bycanal) == len(set(target_depc.keys())):
        print(fixed_bycanal, file = logf)
        print(fixed_desired, file = logf)
        print('# ===== Final Result (Type1.2) ===== #', file = logf)
        print('Correct, all desired nodes are in the canalized region.', file = logf)
        logf.close()
        return 'type2'   
    else:
        return 'pass'

def break_types(expG, considering_fbl, error, iter_n, logf):

    if error: 
        print('# ===== Final Result (Type2.1) ===== #', file = logf)
        print('Error', file = logf)
        logf.close()
        return True

    # threshold
    if len(considering_fbl)>1000:
        print('# ===== Final Result (Type2.2) ===== #', file = logf)
        print('Stop: There are too many FBLs (>1000) at that condition.', file = logf)
        logf.close()
        return True
        
    # splited
    initially_contradicted = np.any([is_contradict(fbl_state) for fbl_state in considering_fbl])
    is_splited = not nx.is_connected(expG.to_undirected()) 
    if initially_contradicted & is_splited & (iter_n>1):
        print('# ===== Final Result (Type2.3) ===== #', file = logf)
        print('Stop: There is a potential to have multiple steady states but cannot determine the condition using motifs.', file = logf)
        logf.close()
        return True

    else:  return False
  
    
    
    
def perturbation_single(model_file, primes, pair, pert_i, pert_str, init_state, target_depc_ori, node_fbls):
    start = time.time()
    logic_reduced_primes, fix_dict_reduced, logic_reduced = reduce_primes(primes, pert_i) 

    iter_n = 0
    reduced_num = 0
    logf = open('./resultLOG/'+model_file.split('.bnet')[0].split('/')[-1]+str(pair)+'_'+str(pert_str)+'.txt','w')
    while 1:
        
        
        # Setting ===================================================================================================================
        init_dict = {x:y for x,y in str2dict(init_state, primes).items() if x in (set(fix_dict_reduced.keys()) | set(logic_reduced_primes.keys()))}
        target_depc = {x:y for x,y in target_depc_ori.copy().items() if x in (set(fix_dict_reduced.keys()) | set(logic_reduced_primes.keys()))}
        node_fbl = [x for x in node_fbls.copy() if (set(x)&(set(fix_dict_reduced.keys()) | set(logic_reduced_primes.keys())) != set())]

        iter_n += 1
        reduced_num += 1

        # a. Check whether the nodes are in the canalized(LDOI) region of {pert_i} ==================================================
        fixed_bycanal = sorted([(x,y) for x,y in fix_dict_reduced.items() if x in target_depc.keys()])
        fixed_desired = sorted([(x,y) for x,y in target_depc.items() if x in fix_dict_reduced.keys()])  
        check_ = is_in_canalizedR(fixed_bycanal, fixed_desired, target_depc, logf)        
        if check_ == 'type1': return ('Wrong','-')
        if check_ == 'type2': return ('Correct','Same')      

        # b. Check is_init_fixed =====================================================================================
        # output: remained_fbl_dict == fix the motif only by inital state (first)!!    
        reduced_bnet = '\n'.join(logic_reduced)
        init_dict_p = init_dict.copy()
        for pi, vii in pert_i.items():
            init_dict_p[pi] = vii
        init_str_p = [pert_dict2str({k:v}) for k,v in init_dict_p.items()]
                
        reduced_bnet_file_name, remained_fbl_dict, remained_fbl_dict_info, modality_error, fixed_canal_reduced = remained_(reduced_bnet, reduced_num, node_fbl, pert_str, init_str_p) # is_init_satisfied
        
        pen = PlayWithExpNet(reduced_bnet_file_name)
        expG = pen.expanded_graph     
        is_notConsidered = [x for x,y in remained_fbl_dict.items() if y['is_notConsidered'][0]]
        considering_fbl = list(itertools.chain(*[x['ssm']+x['csm'] for x in remained_fbl_dict.values() if x['ssm']+x['csm'] != []]))
        if break_types(expG, considering_fbl, modality_error, iter_n, logf): return ('Wrong','-')

        print('# ===== Algorithm 1 ===== #', file = logf)
        print(list(itertools.chain(*[x['ssm']+x['csm'] for x in remained_fbl_dict.values() if x['ssm']+x['csm'] != []])), file = logf)        


        # c. Update stabilized motif =====================================================================================    
        # c-i. check the hierachy btw motifs =============================================================================
        # INPUT: remained_fbl_dict == fix the motif only by inital!!  
        # iteratively confirm the motif state until there is no change!!!
        # OUTPUT: fixed_dict_by_inits        

        def update_stabilzied_motif(remained_fbl_dict, remained_fbl_dict_info):
            fixed_dict_by_inits2 = {}
            while 1:
                fixed_dict_by_inits = {x:defaultdict(set) for x in remained_fbl_dict.keys()}            
                for k,v in remained_fbl_dict.items():
                    ssm_fbl = remained_fbl_dict_info[k]['ssm']
                    csm_fbl = remained_fbl_dict_info[k]['csm']
                    
                    # (1) SSM
                    init_fixedssm = set(itertools.chain(*[list(itertools.chain(*[cn.split('&') for cn in ssm_ ])) for ssm_ in v['ssm']]))
                    if is_contradict(init_fixedssm):
                        print('# ===== Final Result (Type3.1) ===== #', file = logf)
                        print('There would be multiple steady states.', file = logf)
                        logf.close()
                        return ('end') 
                    else:
                        fixed_dict_by_inits[k]['ssm'] = fixed_dict_by_inits[k]['ssm'] | set(v['ssm'])
                                
                    # (2) CSM  
                    for csmi, fbli in itertools.product(v['csm'], csm_fbl+ssm_fbl):               
                        case1, case2, _, same = check_hierarchy_btw_components(primes, expG, pert_i, fixed_canal_reduced, csmi, fbli, init_dict)
                        if (case1==True)&(case2==False)&(same==False): fixed_dict_by_inits[k]['csm'] = fixed_dict_by_inits[k]['csm'] | set([fbli]) 
                        elif (case1==False)&(case2==True)&(same==False): fixed_dict_by_inits[k]['csm'] = fixed_dict_by_inits[k]['csm'] | set([csmi])
                        elif (case1==case2==False)&(same==False): fixed_dict_by_inits[k]['csm'] = fixed_dict_by_inits[k]['csm'] | set([csmi])
                        elif (case1==case2==True)&(same==False):
                            overlap_potent1 = (set([n[1:] if '~' in n else n for n in csmi]) & set(itertools.chain(*is_notConsidered)) != set())
                            overlap_potent2 = (set([n[1:] if '~' in n else n for n in fbli]) & set(itertools.chain(*is_notConsidered)) != set())
                            overlap_p = overlap_potent1 & overlap_potent2
                            fbl2 = set(sorted([x for x in csmi if '&' not in x])) | set(sorted([x for x in fbli if '&' not in x]))
                            if (overlap_p == True) & ~is_contradict(fbl2):
                                fixed_dict_by_inits[k]['csm'] = fixed_dict_by_inits[k]['csm'] | set([fbli]) | set([csmi])                                
                                # print('There is a potential to be stable by integrating feedback motifs')
                            elif (overlap_p == True) & is_contradict(fbl2):
                                print('# ===== Final Result (Type3.2) ===== #', file = logf)
                                print('There would be multiple steady states or oscillation.', file = logf)
                                logf.close()
                                # print('Wrong, There would be multiple steady states or oscillation.') 
                                return ('end')                                     
                            else: 
                                print('# ===== Final Result (Type2.4) ===== #', file = logf)
                                print('Stop: There is insufficient information to discriminate the properties of motifs.', file = logf)
                                logf.close()
                                return ('end') 
                            
                        else: # same 
                             continue
                    fixed_dict_by_inits[k]['ssm_stabilized'] = v['ssm_stabilized']

                if fixed_dict_by_inits == fixed_dict_by_inits2:
                    return (fixed_dict_by_inits2)
                else:
                    remained_fbl_dict = fixed_dict_by_inits
                    fixed_dict_by_inits2 = copy.deepcopy(fixed_dict_by_inits)
                    continue
                
        check_fbl = update_stabilzied_motif(remained_fbl_dict, remained_fbl_dict_info)
        if check_fbl == 'end':
            return ('Wrong','-')
        
        
        fixed_dict_by_inits = check_fbl
        fixed_dict_by_inits = {x:y for x,y in fixed_dict_by_inits.items() if np.any([y['ssm'] != set(), y['csm'] != set()])}
        fixed_ssm = set()
        init_csm = set()
        for values in fixed_dict_by_inits.values():
            fixed_ssm = fixed_ssm | values['ssm'] | set([x for x in values['csm'] if is_ssm(x)])
            init_csm = init_csm | (values['csm']-fixed_ssm)
               
        print('# ===== Algorithm 2 ===== #', file = logf)
        print(list(fixed_ssm | init_csm), file = logf)        


        # c-ii. Extend stabilized motif =========================================================================================
        # functional ssm & init_csm_updated 
        # INPUT: fixed_dict_by_inits
        check_ldoi = defaultdict(list)
        for fbl_cand in (fixed_ssm | init_csm) :
            _, fix_dict_reduced, _ = reduce_primes(logic_reduced_primes, merge_dict(pert_str2dict_list(itertools.chain(*[y.split('&') for y in fbl_cand])))) 
            check_ldoi[fbl_cand] = [pert_dict2str({x[0]:x[1]}) for x in fix_dict_reduced.items()]
            
        # (1) update fixed_ssm
        remained_ssm = set(itertools.chain(*[v['ssm'] for v in remained_fbl_dict_info.values()])) - fixed_ssm        
        for ssm_cand in remained_ssm:
            if (set([x for x in ssm_cand if '&' not in x]) - (set(itertools.chain(*check_ldoi.values())) | set(init_str_p))) == set():
                fixed_ssm = fixed_ssm | set([ssm_cand])
                
        for fbl_cand in remained_ssm:
            _, fix_dict_reduced, _ = reduce_primes(logic_reduced_primes, merge_dict(pert_str2dict_list(itertools.chain(*[y.split('&') for y in fbl_cand])))) 
            check_ldoi[fbl_cand] = [pert_dict2str({x[0]:x[1]}) for x in fix_dict_reduced.items()]  

        
        remained_ssm -= fixed_ssm 
        functional_ssm = fixed_ssm
        # =======================================================
        all_ssmfixed_ldoi = set(itertools.chain(*[check_ldoi[fbl_cand] for fbl_cand in fixed_ssm]))
        if is_contradict(all_ssmfixed_ldoi):
            print('# ===== Final Result (Type3.1) ===== #', file = logf)
            print('There would be multiple steady states', file = logf)
            logf.close()
            # print('Wrong, There would be multiple steady states')  # more than 2
            return('Wrong','-')
        
        
        # =======================================================
        # exception
        if (time.time()-start) > 100000:
            print('Time over')
            print('# ===== Final Result (Type2.5) ===== #', file = logf)
            print('Time over', file = logf)
            logf.close()
            return ('Wrong','-')

        # (2) update csm
        fixed_dictfs = {x:defaultdict(set) for x in fixed_dict_by_inits.keys()}
                        
        for ki in fixed_dictfs.keys():
            vi = remained_fbl_dict_info[ki]
            
            fssm = set(vi['ssm'] + vi['csm']) & functional_ssm
            if fssm != set():
                fixed_dictfs[ki]['ssm'] = fssm
                          
            fssm_csm = set()
            for csm_cand in vi['csm']:     
                    
                # 1) csm update
                all_ifixed_ldoi = set(itertools.chain(*[check_ldoi[fbl_cand] for fbl_cand in (fixed_ssm | init_csm - set([csm_cand]))]))                      
                if (set(itertools.chain(*[x.split('&') for x in csm_cand])) - all_ifixed_ldoi == set()):                   
                    fixed_dictfs[ki]['csm'] = fixed_dictfs[ki]['csm'] | set([csm_cand])
        

                # 2) functional_ssm in csm:                     
                ssmfixed_ldoi = set(itertools.chain(*[check_ldoi[fbl_cand] for fbl_cand in fixed_ssm if not is_contradict(fbl_cand + csm_cand)]))
                compo_cand = [x.split('&') for x in csm_cand if '&' in x]
                
                # a. condpart_connected_by_ssm: condition part is connected by ssm
                condpart_connected_by_ssm = np.all([set(cond)-set(csm_cand)-ssmfixed_ldoi == set() for cond in compo_cand])
                if condpart_connected_by_ssm == False: 
                    continue
                
                
                # b. condpart_fixed: by ssm or initial condition
                condpart_fixed1 = np.all([set(cond)-ssmfixed_ldoi == set() for cond in [x.split('&') for x in csm_cand if '&' in x]])
                condpart_fixed2 = np.all([np.all([is_init_satisfied(expG, c2, fixed_canal_reduced, init_str_p) for c2 in set(cond)&set(csm_cand)]) for cond in compo_cand])
                condpart_fixed = condpart_fixed1 | condpart_fixed2
                if condpart_fixed == False: 
                    continue
                                                

                # c. is_no_connection // check_hierachy
                other_case = []
                for other in vi['ssm']+vi['csm']:
                    case1, case2, _, same = check_hierarchy_btw_components(primes, expG, pert_i, fixed_canal_reduced, csm_cand, other, init_dict)
                    if (case1==False)&(same==False):
                        other_case.append(True)
                    elif (same == False):
                        other_case.append(False)
                    else:
                        continue
      
                condition_fixed_case = condpart_connected_by_ssm & condpart_fixed & np.all(other_case) & (other_case != [])
                if condition_fixed_case: fssm_csm = fssm_csm | set([csm_cand])

                            
            fixed_dictfs[ki]['csm'] = fixed_dictfs[ki]['csm'] | fssm_csm
            functional_ssm = functional_ssm | fssm_csm
            

        # no update
        fixed_fs = {x:y for x,y in fixed_dictfs.items() if np.any([y['ssm'] != set(), y['csm'] != set()])}
        for ori_key in fixed_dictfs.keys():
            if ori_key not in fixed_fs.keys():
                fixed_fs[ori_key] = fixed_dict_by_inits[ori_key]
        
               
        print('# ===== Algorithm 3 ===== #', file = logf)
        print(set(functional_ssm), file = logf)        
        


        all_funcssmfixed_ldoi = set(itertools.chain(*[check_ldoi[fbl_cand] for fbl_cand in functional_ssm]))
        if is_contradict(all_funcssmfixed_ldoi):
            print('# ===== Final Result (Type3.1) ===== #', file = logf)
            print('There are multiple steady states.', file = logf)
            logf.close()
            print('Wrong1, There are multiple steady states')  # more than 2
            return ('Wrong','-')
        
        
        # =========================================================================================
        # end
        return_loops = []            
        for value_dict in fixed_fs.values():
            value_cand = value_dict['ssm'] | value_dict['csm']
            csm_cand_set = value_cand-functional_ssm
            remained_csm_cands = set(itertools.chain(*[[x for x in cx  if '&' not in x] for cx in csm_cand_set]))
            if is_contradict(remained_csm_cands):
                for csm_comb in list(itertools.combinations(csm_cand_set,2)):
                    if np.any([nx.has_path(expG, pair[0], pair[1]) for pair in itertools.product(csm_comb[0], csm_comb[1])]):
                        print(len(expG.nodes))
                        return_loops.append(True)
                        print('There is a connection btw the motifs')
                        break
                    else: 
                        return_loops.append(False)
                        print('There are unconnected contradictory motifs')
                        break
            else: 
                return_loops.append(False)

        return_loop = np.any(return_loops)


        # (4-3) =========================================================================================

        final_fblstates = set(functional_ssm) | set(itertools.chain(*[x['ssm']| x['csm'] for x in fixed_fs.values() if (x['ssm'] | x['csm']) != set([])]))
        final_fbls = defaultdict(list)
        for fblstate in final_fblstates:
            key = tuple(set(itertools.chain(*[sorted([y[1:] if '~' in y else y for y in n.split('&')]) for n in fblstate])))
            final_fbls[key] = fblstate
    
        # fix!!
        fixbyfuncSSM = set(itertools.chain(*[check_ldoi[fbl_cand] for fbl_cand in functional_ssm]))
        fix_node = set([x if '~' not in x else x[1:] for x in fixbyfuncSSM])
        
        fixed_str = fixbyfuncSSM | set(itertools.chain(*[check_ldoi[final_fbls[fbl_cand]] for fbl_cand in set(final_fbls.keys()) if set(fbl_cand)&fix_node == set()]))        
        fixed_wi = pert_i.copy()
        fixed_wi.update(merge_dict(pert_str2dict_list(fixed_str))) 
        _, fixed_dict, _ = reduce_primes(primes, fixed_wi) 
        

        print('# ===== Final fixed state ===== #', file = logf)
        print(set([pert_dict2str({k:v}) for k,v in fixed_dict.items()]), file = logf)        
   
        

        # =========================================================================================
        if set(target_depc.items()) - set(fixed_dict.items()) != set():
            return_loop = True         
        # =========================================================================================
        
        
        if return_loop == True:
            if (time.time()-start) > 10000:
                print('Time over')
                print('# ===== Final Result (Type2.5) ===== #', file = logf)
                print('Time over', file = logf)
                
                logf.close()
                return ('Wrong','-')
   
            

            pert_i.update(merge_dict(pert_str2dict_list(fixbyfuncSSM))) # wo csm info. 
            logic_reduced_primes, fix_dict_reduced, logic_reduced_c = reduce_primes(primes, pert_i)
            if logic_reduced == logic_reduced_c:
                #print('4-3 or 4-4, There are no other cases on this network') 
                print('# ===== Final Result (Type2.4) ===== #', file = logf)
                print(sorted(fixed_dict.items()), file = logf)
                print('unknown_depc:',set(target_depc.items()) - set(fixed_dict.items()), file=logf)
                logf.close()
                if (set(target_depc.items()) - set(fixed_dict.items()) == set()): return('Correct','Less')
                else: return('Wrong','-')
            else:
                logic_reduced = logic_reduced_c
                continue
            
            
        else:
            print('# ===== Final Result ===== #', file = logf)
            print(sorted(fixed_dict.items()), file = logf)
            print('unknown_depc:',set(target_depc.items()) - set(fixed_dict.items()), file=logf)
            logf.close()
            if (set(target_depc.items()) - set(fixed_dict.items()) == set()): return('Correct','Less')
            else: return('Wrong','-')                                  

