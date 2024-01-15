# -*- coding: utf-8 -*-
"""

@author: Jonghoon Lee
"""

from collections import deque
import networkx as nx
import pandas as pd

import os
from modules.qm import QM # Quine-McCluskey algorithm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class PlayWithExpNet(object):
	def __init__(self, boolnet_file, fixed={}, prefix='n', suffix='n', composite_delim='&'):
		self._prefix = prefix
		self._suffix = suffix
		self._composite_delim = composite_delim
		self._load_model(boolnet_file)
		self._expand_network()
	
	def _load_model(self, boolnet_file):
		# Read the original graph
		with open(boolnet_file, 'r') as f:
			lines = f.readlines()
		original_graph, original_nodes = form_network(lines, sorted_nodename=False)
		self.original_graph = original_graph
		self.original_nodes = original_nodes
		
		# Map of Node-to-internal_index
		self._internal_to_node = {}
		self._node_to_internal = {}
		for index, node_name in zip([i for i in range(len(original_nodes))], original_nodes):
			internal_name = self._prefix + str(index) + self._suffix
			internal_name_neg = '~' + internal_name
			node_name_neg = '~' + node_name
			self._internal_to_node[internal_name] = node_name
			self._internal_to_node[internal_name_neg] = node_name_neg
			self._node_to_internal[node_name] = internal_name
			self._node_to_internal[node_name_neg] = internal_name_neg
	
	def _expand_network(self):
		if not self.original_graph:
			raise ValueError("Load network file!")
		# Genearate expanded network
		self._expanded_graph_internal = Get_expanded_network(self.original_graph, prefix=self._prefix, suffix=self._suffix, composite_delim=self._composite_delim)
		self.expanded_graph = self._convert_from_internal(self._expanded_graph_internal)

	def _convert_from_internal(self, graph):
		converted_graph = nx.DiGraph()
		i_to_n = {}
		n_to_i = {}
		for inode_name in graph.nodes:
			inode_list = [x for x in inode_name.split(self._composite_delim)]
			# print(inode_list)
			cnode_name = self._composite_delim.join([self._internal_to_node[inode] for inode in inode_list])


			converted_graph.add_node(cnode_name)
			i_to_n[inode_name] = cnode_name
			# n_to_i[cnode_name] = inode_name

		for parent, child in graph.edges:
			converted_graph.add_edge(i_to_n[parent], i_to_n[child])

		return converted_graph

	def get_expanded_network(self):
		return self.expanded_graph
	
def perturb_expanded_network(expanded_graph, node_state_map, composite_delim='&'):
	# Check error
	if not isinstance(node_state_map, dict):
		raise ValueError("")
	for node_name in node_state_map.keys():
		if node_name in expanded_graph.nodes:
			continue
		if composite_delim not in node_name:
			continue
		raise ValueError("Invalid node name: " + node_name)
	
	expanded_graph = nx.DiGraph(expanded_graph)
	
	for node_name, state in node_state_map.items():
		fix_node(expanded_graph, node_name, state)
	
	return expanded_graph


def fix_node(enet, node_name, state=False, composite_delim="&"):
	node_name = node_name if state else '~' + node_name
	neg_node_name = Negation_in_expanded(node_name)

	# Remove the negate node and its corresponding composite nodes
	remove_nodes = []
	for node in enet.nodes:
		if neg_node_name not in node.split(composite_delim):
			continue
		remove_nodes.append(node)
	enet.remove_nodes_from(remove_nodes)

	# Remove the in-links of perturbed node
	remove_edges = []
	for edge in enet.edges:
		# if node_name not in edge[1].split(composite_delim):
		if node_name != edge[1]:
			continue
		remove_edges.append(edge)
	enet.remove_edges_from(remove_edges)
	
	# Remove unnecessary composite node
	remove_nodes = [x for x in enet.nodes() if (enet.out_degree(x) == 0) and (composite_delim in x)]
	enet.remove_nodes_from(remove_nodes)
	
	return enet


def to_sif(graph, file_prefix, composite_delim='&'):
	node_tb = pd.DataFrame(columns=['Node', 'Type'])
	for node in graph.nodes:
		# print('Single' if composite_delim not in node else 'Composite')
		node_tb = node_tb.append({'Node': node,
								  'Type': 'Single' if composite_delim not in node else 'Composite'}, ignore_index=True)
	
	edge_tb = pd.DataFrame(columns=['Source', 'Target'])
	for edge in graph.edges:
		# print('Single' if composite_delim not in node else 'Composite')
		edge_tb = edge_tb.append(
			{'Source': edge[0], 'Target': edge[1]}, ignore_index=True)
	
	node_tb.to_csv(file_prefix + '_node.tsv', index=False, header=True, sep='\t')
	edge_tb.to_csv(file_prefix + '_edge.tsv', index=False, header=True, sep='\t')
	

'''
	From BooleanDOI
'''
def form_network(rules, sorted_nodename=True):
	'''
	Takes as input a list of rules in the format of Booleannet(e.g. sample_network.txt)

	Outputs a networkx DiGraph with node properties:
		'update_nodes': a list of regulating nodes
		'update_rules': a dictionary with binary strings as keys, corresponding
						to the possible states of the update_nodes, and integers
						as values, corresponding to the state of the node given
						that input.

	Note that nodes are identified by their position in a sorted list of
	user-provided node names.

	Notice the name of one node should not be part of other node

	The code is written by Colin Campbell.
	'''
	
	def clean_states(x):
		# cleans binary representation of node input states
		out = x[2:]  # Strip leading 0b
		return '0' * (len(inf) - len(out)) + out  # Append leading 0's as needed
	
	stream = [x.rstrip('\n') for x in rules if x != '\n' and x[0] != '#']  # Remove comments and blank lines
	# I made a slight change here so that the code will be compatible with Jorge's Java code
	# Generate a sorted list of node names
	if sorted_nodename:
		nodes = sorted([x.split('*=', 1)[0].strip() for x in stream])  # 2020.10.19: added strip() by Jonghoon Lee
	else:
		nodes = [x.split('*=', 1)[0].strip() for x in stream]  # 2020.10.19: added strip() by Jonghoon Lee
	
	g = nx.DiGraph()
	g.graph['knockout'] = None  # At creation, no node is flagged for knockout or overexpression
	g.graph['express'] = None
	
	for n in range(len(stream)):
		node = stream[n].split('*=')[0].strip()
		rule = stream[n].split('*=')[1].strip()
		rule = rule.replace(' AND ', ' and ')  # Force decap of logical operators so as to work with eval()
		rule = rule.replace(' OR ', ' or ')
		rule = rule.replace(' NOT ', ' not ')
		if stream[n].find('True') >= 0 or stream[n].find('False') >= 0:  # For always ON or always OFF nodes
			g.add_node(nodes.index(
				node))  # We refer to nodes by their location in a sorted list of the user-provided node names
			g._node[nodes.index(node)]['update_nodes'] = []
			g._node[nodes.index(node)]['update_rules'] = {'': str(int(eval(rule)))}
			continue
		
		inf = rule.split(' ')  # Strip down to just a list of influencing nodes
		inf = [x.lstrip('(') for x in inf]
		inf = [x.rstrip(')') for x in inf]
		# The sort ensures that when we do text replacement (<node string>->'True' or 'False') below in this fn, we avoid problems where node 1 is a substring of node 2 (e.g. NODE1_phosph and NODE1)
		inf = sorted([x for x in inf if x not in ['', 'and', 'or', 'not']], key=len, reverse=True)
		inf = [x for x in set(inf)]  # 04/16/2016 to allow one variable appear twice in the rule like a XOR rule
		
		# mod
		for i in inf:
			g.add_edge(nodes.index(i), nodes.index(node))  # Add edges from all influencing nodes to target node
		
		g._node[nodes.index(node)]['update_nodes'] = [nodes.index(i) for i in inf]  # mod
		g._node[nodes.index(node)]['update_rules'] = {}  # mod
		
		bool_states = map(bin, range(2 ** len(inf)))
		bool_states = map(clean_states, bool_states)
		for j in bool_states:
			rule_mod = rule[:]
			for k in range(len(j)):
				# Modify the rule to have every combination of True, False for input nodes
				if j[k] == '0':
					rule_mod = rule_mod.replace(nodes[g._node[nodes.index(node)]['update_nodes'][k]], 'False')
				else:
					rule_mod = rule_mod.replace(nodes[g._node[nodes.index(node)]['update_nodes'][k]], 'True')
			
			# Store outcome for every possible input
			g._node[nodes.index(node)]['update_rules'][j] = int(eval(rule_mod))
		# mod
	
	return g, nodes


'''
	From BooleanDOI
'''
def Getfunc(inputlist, output, onlist, prefix='n', suffix='n', equal_sign='*='):
	'''
	Return a Boolean regulatory rule in the disjuctive normal form (in the format of Booleannet)
	based on given truth table.
	For more details about Booleannet, refer to https://github.com/ialbert/booleannet.
	It is essentially a wrapper for the Quine-McCluskey algorithm to meet the format of Booleannet.

	Parameters
	----------
	inputlist : a list of nodenames of all the parent nodes/ regulators
	output    : the nodename of the child node
	onlist    : a list of all configurations(strings) that will give a ON state of the output based on rule
				e.g. the onlist of C = A OR B will be ['11','10','01']
	prefix='n': prefix to encode the node name to avoid one node's name is a part of another node's name
	suffix='n': suffix to encode the node name to avoid one node's name is a part of another node's name
				e.g. node name '1' will become 'n1n' in the returned result
	equal_sign: the equal sign of the rule in the returned result, whose default value follows the Booleannet format
 
	Returns
	-------
	The Boolean rule in the disjuctive normal form.
 
 
	References
	----------
	QM code by George Prekas.
	'''
	# pre-processing to meet the format requirement of QM and Booleannet
	inputs = [prefix + str(x) + suffix for x in inputlist]
	inputs.reverse()
	output = prefix + str(output) + suffix + equal_sign
	onindexlist = [int(x, 2) for x in onlist]
	if len(inputs) > 0:
		qmtemp = QM(inputs)
		temprule = qmtemp.get_function(qmtemp.solve(onindexlist, [])[1])
		temprule = temprule.replace('AND', 'and')
		temprule = temprule.replace('OR', 'or')
		temprule = temprule.replace('NOT', 'not')
	else:
		temprule = output[:-2]
	return output + temprule + '\n'


'''
	From BooleanDOI
'''
# 2020.10.19 : add arguments : composite_delim
def Get_expanded_network(Gread, prefix='n', suffix='n', equal_sign='*=', composite_delim='_'):
	'''
	Return the expanded network for a given Boolean network model.
	The Boolean network model is a DiGraph object in the output format of form_network().
	The Boolean network model can be generated through form_network function by reading a text file in the Booleannet format.
	The Boolean rules will first be converted to a disjuctive normal form before generating the expanded network.
 
	Parameters
	----------
	Gread     : the given Boolean network model
	prefix='n': prefix to encode the node name to avoid one node's name is a part of another node's name
	suffix='n': suffix to encode the node name to avoid one node's name is a part of another node's name
				e.g. node name '1' will become 'n1n' in the returned result
	equal_sign: the equal sign of the rule in the returned result, whose default value follows the Booleannet format
 
	Returns
	-------
	The expanded network for the given Boolean network model.
 
 
	'''
	G_expand = nx.DiGraph()
	rules = []
	# first write rules for negation nodes
	negation_rules = []
	expanded_nodes = set()
	for node in Gread.nodes():
		ON_list = [x for x in Gread._node[node]['update_rules'].keys() if Gread._node[node]['update_rules'][x] == 1]
		rule = Getfunc(Gread._node[node]['update_nodes'], node, ON_list, prefix=prefix, suffix=suffix,
					   equal_sign=equal_sign)
		OFF_list = [x for x in Gread._node[node]['update_rules'].keys() if Gread._node[node]['update_rules'][x] == 0]
		negation_rule = '~' + Getfunc(Gread._node[node]['update_nodes'], node, OFF_list, prefix=prefix, suffix=suffix,
									  equal_sign=equal_sign)
		# print(node, rule)
		rules.append(rule)
		negation_rules.append(negation_rule)
		expanded_nodes.add(rule.split('*=')[0])
		expanded_nodes.add(negation_rule.split('*=')[0])
	# then for each line in the rules, construct Boolean network
	composite_nodes = []
	rules.extend(negation_rules)
	for line in rules:
		child, update_rule = line.split('*=')
		update_rule = update_rule.strip()
		if update_rule[0] == '(' and update_rule[-1] == ')':
			update_rule = update_rule[1:-1]
		# single parent situation
		if child[0] == '~':
			normal_child = child[1:]
		else:
			normal_child = child[:]
		normal_child = normal_child[len(prefix):len(normal_child) - len(suffix)]
		# deal with source node situation
		if not Gread._node[int(normal_child)]['update_nodes']:
			G_expand.add_node(child)  # maybe this do not need to be done
		else:
			if 'or' in update_rule:
				parents = update_rule.split(' or ')
			else:
				parents = [update_rule]
			parents.sort()
			for parent in parents:
				parent = parent.replace('not ', '~').replace('(', '').replace(')', '')
				if 'and' in parent:
					composite_node = parent.replace(' and ', composite_delim)
					composite_nodes.append(composite_node)
					G_expand.add_edge(composite_node, child)
					for component in composite_node.split(composite_delim):
						G_expand.add_edge(component, composite_node)
				else:
					G_expand.add_edge(parent, child)
	return G_expand.copy()


'''
	From BooleanDOI
'''
import modules.BooleanDOI_TargetControl as BDOItc
# 2022.07.27 : modify 
def update_single_DOI(G_expanded, result_filename, TDOI_BFS, flagged, potential, composite_delim='_'):
    '''
    The pre-processing step of the GRASP algorithm to solve the target control problem.
    The function calculates the LDOI, whether this node is an incompatible intervention and composite nodes attached to the LDOI
    as recorded in TDOI_BFS, flagged, potential
    This function also output three files to record the above three quantites.
    The three files share the first part of the filename as result_filename.
    
    Parameters
    ----------
    G_expanded : the input expanded_graph as a DiGraph object
    result_filename : the prefix of the output file names
    TDOI_BFS : a dictionary to store the LDOI of each node state set
    flagged : a dictionary to store whether the LDOI of each node state set contains the negation of any target node
    potential : a dictionary to store the composite nodes visited but not included during the search process for LDOI for each node state set
    
    Returns
    -------
    None
    '''
    checklist = [result_filename+suffix for suffix in ['_TDOI.txt', '_flagged.txt', '_potential.txt']]
    TDOI_filename, flagged_filename, potential_filename = checklist
    # result_exist= all([os.path.isfile(item) for item in checklist])
    # if not result_exist:

    normal_nodes=[node for node in G_expanded.nodes() if composite_delim not in node]
    for node in normal_nodes:
        BDOItc.update_DOI(G_expanded,[node],TDOI_BFS,flagged,potential)
    BDOItc.write_node_property(normal_nodes,TDOI_filename,TDOI_BFS)
    BDOItc.write_node_property(normal_nodes,flagged_filename,flagged)
    BDOItc.write_node_property(normal_nodes,potential_filename,potential)
    # else:
    #   BDOItc.read_TDOI(TDOI_filename,TDOI_BFS)
    #   BDOItc.read_flagged(flagged_filename,flagged)
    #   BDOItc.read_potential(potential_filename,potential)
    return


'''
	From BooleanDOI
'''
def Negation(node):
  '''
  Return the complementary node of a given node in the expanded network generated by Jorge's program.
  That is take the negation of the given node.
  This does not apply to composite node.
  '''
  if '.' not in node:
    return str(-int(node))
  else:
    return str(-float(node))

'''
	From BooleanDOI
'''
def Negation_in_expanded(node):
  '''
  Return the complementary node of a given node in the expanded network.
  That is take the negation of the given node.
  This does not apply to composite node.
  '''
  if '~' in node:
    return node[1:]
  else:
    return '~'+node

