    
import os, time
import sys
sys.path.append("")
import unittest
# import hashlib
import numpy as np
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import chisq, fisherz, gsq, kci, mv_fisherz, d_separation
from causallearn.graph.SHD import SHD
from causallearn.utils.DAG2CPDAG import dag2cpdag
from causallearn.utils.TXT2GeneralGraph import txt2generalgraph
from tests.utils_simulate_data import simulate_discrete_data, simulate_linear_continuous_data

# print('Now start test_pc_simulate_linear_nongaussian_with_kci ...')
# print('!! It will take around 17 mins to run this test (on M1 Max chip) ... !!')
# print('!! You may also reduce the sample size (<2500), but the result will then not be totally correct ... !!')

# Graph specification.
# num_of_nodes = 5
# truth_DAG_directed_edges = {(0, 1), (0, 3), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)}
# truth_CPDAG_directed_edges = {(0, 3), (1, 3), (2, 3), (2, 4), (3, 4)}
# truth_CPDAG_undirected_edges = {(0, 1), (1, 2), (2, 1), (1, 0)}
# After the skeleton is discovered, the edges are oriented in the following way:
# Unshilded triples:
#   2 -- 1 -- 0: not v-structure.
#   1 -- 2 -- 4: not v-structure.
#   0 -- 3 -- 2: v-structure, oritented as 0 -> 3 <- 2.
#   0 -- 3 -- 4: not v-structure.
#   1 -- 3 -- 4: not v-structure.
# Then by Meek rule 1: 3 -> 4.
# Then by Meek rule 2: 2 -> 4.
# Then by Meek rule 3: 1 -> 3.

# truth_DAG_directed_edges = {(0, 2), (0, 3), (0, 4), (2, 4), (2, 3), (3, 4), (1, 2), (1, 4)}
truth_DAG_directed_edges = {(0, 2), (1, 2)}#, (3, 2)}
num_of_nodes = max(sum(truth_DAG_directed_edges, ())) + 1

data = simulate_discrete_data(num_of_nodes, 10000, truth_DAG_directed_edges, 42)
## there is no randomness in data generation (with seed fixed for simulate_data).
## however, there still exists randomness in KCI (null_sample_spectral).
## for this simple test, we can assume that KCI always returns the correct result (despite randomness).

# Run PC with default parameters: stable=True, uc_rule=0 (uc_sepset), uc_priority=2 (prioritize existing colliders)
# cg = pc(data=data, alpha=0.05, stable=True, uc_rule=0, uc_priority=-1, show_progress=True, indep_test='gsq')
# Run PC with: stable=True, uc_rule=0 (uc_sepset), uc_priority=0 (overwrite)
cg = pc(data, 0.1, ikb=False, uc_rule=1, uc_priority=4)
cg.draw_pydot_graph()


###### Simulation configuration: code to generate "./TestData/test_pc_simulated_linear_gaussian_data.txt" ######
# data = simulate_linear_continuous_data(num_of_nodes, 10000, truth_DAG_directed_edges, "gaussian", 42)
###### Simulation configuration: code to generate "./TestData/test_pc_simulated_linear_gaussian_data.txt" ######

# this simple graph is the same as in test_pc_simulate_linear_gaussian_with_fisher_z.

# from cdt.data import load_dataset
# data, graph = load_dataset("sachs")
# data = data.to_numpy()

### binary dataset
# import bnlearn as bn
# bif_file= 'sachs'
# data = bn.import_DAG(bif_file, verbose=1).to_numpy()
# def checkEqual(L1, L2, output_message):
#     # if type(L1) != list:
#     #     L1 = list(L1)
#     #     L2 = list(L2)

#     if set(L1)!=set(L2):
#         return print(output_message)       

# data = simulate_linear_continuous_data(num_of_nodes, 2500, truth_DAG_directed_edges, "gaussian", 42)
## there is no randomness in data generation (with seed fixed for simulate_data).
## however, there still exists randomness in KCI (null_sample_spectral).
## for this simple test, we can assume that KCI always returns the correct result (despite randomness).

# Run PC with default parameters: stable=True, uc_rule=0 (uc_sepset), uc_priority=2 (prioritize existing colliders)
# cg = pc(data=data, alpha=0.05, stable=True, uc_rule=0, uc_priority=-1, show_progress=True#, indep_test='kci'
# )
# returned_directed_edges = set(cg.find_fully_directed())
# returned_undirected_edges = set(cg.find_undirected())
# returned_bidirected_edges = set(cg.find_bi_directed())
# checkEqual(truth_CPDAG_directed_edges, returned_directed_edges, "Directed edges are not correct.")
# checkEqual(truth_CPDAG_undirected_edges, returned_undirected_edges, "Undirected edges are not correct.")
# print("There should be no bi-directed edges.") if 0!=len(returned_bidirected_edges) else print("finished")

verb=False

#======================================================================================#
#                                 Define Pearl Axioms                                  #
#======================================================================================#


class test_obj( object ):
    def __init__(self, X:set, S:set, Y:set, p_val:float=None, dep_type:str="I", alpha:float=0.05 ):
        self.X= X
        self.Y= Y
        self.S= S
        self.p_val = p_val
        if p_val == None:
            self.dep_type = dep_type
        else:
            self.dep_type = "D" if p_val < alpha else "I"

    def to_list(self, p=False)->list:
        if p:
            return [self.X, self.S, self.Y, self.p_val, self.dep_type]
        else:
            return [self.X, self.S, self.Y, self.dep_type]

    def elements(self):
        return (self.X, self.S, self.Y)

    def negate(self):
        if self.dep_type=="D":
            self.dep_type="I"
        else:
            self.dep_type="D"
        return self

    def states_independence(self):
        return self.dep_type == "I"

    ## Symmetry
    def symmetrise(self, verbose=verb):
        if self.dep_type=="I":
            return test_obj(X=self.Y, S=self.S, Y=self.X, p_val=self.p_val, dep_type=self.dep_type)

    ## Decomoposition
    def decompose(self, verbose=verb)->list:
        assert len(self.Y)>=2
        return_list=[]
        for item in self.Y:
            return_list.append(test_obj(self.X,self.S,{item}))
        if verbose:
            print("Decompose:",{'P':self.to_list(), 'C':[item.to_list() for item in return_list]})
        return return_list #NOTE: add pval (semantics driven?)

    ## Weak Union
    def weak_union(self, verbose=verb)->list:
        assert len(self.Y)>=2
        Y, W = self.Y
        if verbose:
            print("Weak Union:",{'P':self.to_list(), 'C':[self.X, self.S.union({W}), {Y}, self.dep_type]})
        return [test_obj(self.X, self.S.union({W}), {Y})] #NOTE: add pval


# test=test_obj(X={0},Y={1,4},S={2,3})
# print(f"Initial test: [{test.to_list()}] i.e. {test.X} _||_ {test.Y} | {test.S} ")

# ## Decomposition
# print(f"Apply Decomposition: {test.decompose()}")

# ## Weak Union
# print(f"Apply Weak Union: {test.weak_union()}")

# test1=test_obj(X={0}, Y={1,5}, S={2,3})
# test2=test_obj(X={0}, Y={4}, S={2,1,3,5})
# print(f"Initial tests: {test1.to_list()} & {test2.to_list()}")

## Contraction
def contract(test1:test_obj, test2:test_obj, verbose=verb)->test_obj:
    ##NOTE: need to take care of different lenghts because it then come decomposition
    
    conditions = [len(test1.S)>0 , len(test2.S)>0 , test1.dep_type==test2.dep_type=='I' , test1.X==test2.X , test1.Y!=test2.Y , test1.Y<test2.S, test2.S-test1.Y==test1.S]
    if all(conditions):
        if verbose:
            print("Contraction:", {'P':[test1.to_list(), test2.to_list()],'C':[test1.X, test1.S , test1.Y.union(test2.Y), test1.dep_type]})
        return test_obj(test1.X, test1.S , test1.Y.union(test2.Y))
    # elif o := All(conditions):
            # print(f'Contraction False conditions: {np.array(["len(test1.S)>0" , "len(test1.S)>0" , "test1.dep_type==test1.dep_type==I", "test1.X == test2.X", "test1.Y != test2.Y", "test1.Y.issubset(test2.S)"])[o.index]}')
            # print(f"Failed conditions:{o.index}")
# print(f"Apply Contraction: {contract(test1, test2)}")

def contract_decompose(test1:test_obj, test2:test_obj)->list:
    contracted= contract(test1, test2)
    if type(contracted) == test_obj:
        return contracted.decompose()
    else:
        return contracted

def contract_wunite(test1:test_obj, test2:test_obj)->list:
    contracted= contract(test1, test2)
    if type(contracted) == test_obj:
        return contracted.weak_union()
    else:
        return contracted

# print(f"Apply Decomposition to the Contraction: {contract_decompose(test1, test2)}")

# test1=test_obj(X={0}, Y={1}, S={2,4,3})
# test2=test_obj(X={0}, Y={4}, S={2,1,3})
# print(f"Initial tests: {test1.to_list()} & {test2.to_list()}")

def intersect(test1:test_obj, test2:test_obj, verbose=verb)->list:
    Z = test1.S.intersection(test2.S)
    W = test1.S - Z
    Y = test2.S - Z
    # if len(test1.S)==len(test1.S)>0 and test1.dep_type==test1.dep_type=='I' and test1.X == test2.X and Z1==Z2 and test1.Y.issubset(test2.S):
    #     return {'P':[test1.to_list(), test2.to_list()],'C':[test1.X, Z1 , test1.Y.union(test2.Y), test1.dep_type]}
    conditions = [len(test1.S)==len(test2.S)>0 , test1.dep_type==test2.dep_type=='I' , test1.X == test2.X , W==test2.Y, test1.Y==Y , test1.Y < test2.S ]
    if all(conditions):
        if verbose:
            print("Intersection:", {'P':[test1.to_list(), test2.to_list()],'C':[test1.X, Z , test1.Y.union(test2.Y), test1.dep_type]})
        return test_obj(test1.X, Z , test1.Y.union(test2.Y))
    # elif o := All(conditions):
    #         print(f'Intersection False conditions: {np.array(["len(test1.S)==len(test1.S)>0" , "test1.dep_type==test1.dep_type==I" , "test1.X == test2.X" , "Z1==Z2" , "test1.Y.issubset(test2.S)"])[o.index]}')
            # print(f"Failed conditions:{o.index}")

# print(f"Apply Intersection: {intersect(test1, test2)}")

def intersect_decompose(test1:test_obj, test2:test_obj)->list:
    intersected= intersect(test1, test2)
    if type(intersected) == test_obj:
        return intersected.decompose()

def intersect_wunite(test1:test_obj, test2:test_obj)->list:
    intersected= intersect(test1, test2)
    if type(intersected) == test_obj:
        return intersected.weak_union()

# print(f"Apply Decomposition to the Intersection: {intersect_decompose(test1, test2)}")

# test1=test_obj(X={0}, Y={1}, S={2,})
# test2=test_obj(X={0}, Y={1}, S={2,3})
# print(f"Initial tests: {test1.to_list()} & {test2.to_list()}")

#### Additional Rules for Bayesian Networks

def make_lor_single_head(P1:test_obj, P2:test_obj, C1:test_obj, C2:test_obj, verbose=verb)->list:
    R1= {'P':[P1.to_list(),P2.to_list(),C1.negate().to_list()],'C':[C2.to_list()]}
    R2= {'P':[P1.to_list(),P2.to_list(),C2.negate().to_list()],'C':[C1.to_list()]}
    R3= {'P':[P1.to_list(),C1.negate().to_list(),C2.negate().to_list()],'C':[P2.negate().to_list()]}
    R4= {'P':[P2.to_list(),C1.negate().to_list(),C2.negate().to_list()],'C':[P1.negate().to_list()]}
    if verbose:
        print({key: [value] + [R2[key]] + [R3[key]] + [R4[key]] for key, value in R1.items()})
    return [P1,P2,C1.negate(),C2] + [P1,P2,C2.negate(),C1] + [P1,C2.negate(),C1.negate(),P2.negate()] + [P2,C2.negate(),C1.negate(),P1.negate()] 
    #TODO: these will be single rules to be coded separately.

## Weak Transitiviy
def weak_transitivity(test1:test_obj, test2:test_obj, verbose=verb)->dict:
    Z = test1.S.intersection(test2.S)
    W = test2.S - Z
    # if len(test1.S)==len(test1.S)>0 and test1.dep_type==test1.dep_type=='I' and test1.X == test2.X and Z1==Z2 and test1.Y.issubset(test2.S):
    #     return {'P':[test1.to_list(), test2.to_list()],'C':[test1.X, Z1 , test1.Y.union(test2.Y), test1.dep_type]}
    conditions = [len(test1.S)>0, len(test2.S)>0, test1.dep_type==test2.dep_type=='I', test1.X==test2.X, test1.Y==test2.Y, test1.S<test2.S ]
    if all(conditions):
        P00 = test_obj(test1.X, test1.S, test1.Y)
        P01 = test_obj(test2.X, test2.S, test2.Y)
        C00 = test_obj(test1.X, test1.S, W)
        C01 = test_obj(W, test1.S, test1.Y)
        if verbose:
            print("Weak Transitivity:", make_lor_single_head(P00,P01,C00,C01))
        return make_lor_single_head(P00,P01,C00,C01)

# print(f"Apply Weak Transitivity: {weak_transitivity(test1, test2)}")


# test1=test_obj(X={0}, Y={1}, S={2,3})
# test2=test_obj(X={2}, Y={3}, S={0,1})
# print(f"Initial tests: {test1.to_list()} & {test2.to_list()}")

## Chordiality
def chordiality(test1:test_obj, test2:test_obj)->dict:
    assert len(test1.S)==2
    assert len(test2.S)==2
    assert test1.X.union(test1.Y) == test2.S
    assert test2.X.union(test2.Y) == test1.S
    X2, Y2 = test1.S
    P00 = test_obj(test1.X, test1.S, test1.Y)
    P01 = test_obj(test2.X, test2.S, test2.Y)
    C00 = test_obj(test1.X, {X2}, test1.Y)
    C01 = test_obj(test1.X, {Y2}, test1.Y)
    return make_lor_single_head(P00,P01,C00,C01)

# print(f"Apply Chordiality: {chordiality(test1, test2)}")
# print("="*100)

## Composition
def compose(test1:test_obj, test2:test_obj)->list:
    raise NotImplementedError


sys.path.append("../")
# from abap_parser import *
# from aspartix_interface import *
from ABAplus.aba_plus_ import Rule, Sentence, Preference, ABA_Plus, LESS_THAN, LESS_EQUAL, NO_RELATION, CANNOT_BE_DERIVED, NORMAL_ATK, REVERSE_ATK
from ABAplus.aba_plus_ import *
from tqdm.auto import tqdm
import itertools

IKB_axioms = cg.IKB_list.copy()


#======================================================================================#
#                                  Apply Pearl Axioms                                  #
#======================================================================================#

test_assumptions = []
decision_rules = [] 
symmetry_rules = []
contraction_rules = []
symmetry_rules = []
decomposition_rules = []
union_rules = []   
intersection_rules = []
contrary_rules = []
p_preferences = []
test_to_sentence_map = {}

for test in tqdm(cg.IKB_list):
    test_sentence = Sentence(str(test.to_list()))
    test_assumptions.append(test_sentence) if test.p_val != None else test_assumptions ##NOTE: not allowing deducted tests to get into assumptions, should I?
    test_to_sentence_map[test] = test_sentence
    add = test.symmetrise()
    if add != None and add.elements() not in [test.elements() for test in IKB_axioms]:
        IKB_axioms.append(add)
        add_sentence = Sentence(str(add.to_list()))
        test_to_sentence_map[add] = add_sentence
        # test_assumptions.append(add_sentence) ##TODO: should symmetric be in the assumption as well??
        rule = Rule(set([test_sentence]), add_sentence)
        symmetry_rules.append(rule)

test_assumptions=set(test_assumptions)
print("Number of assumptions:", len(test_assumptions))
print("Number of rules (symmetry):", len(symmetry_rules))
print("Tests added by symmetry:",len(IKB_axioms)-len(cg.IKB_list))

##================================================================
### put in a fake contrary test
# fake_test = test_obj(X={0},S=set(),Y={1},p_val=0.1)
# fake_test_sentence = Sentence(str(fake_test.to_list()))
# fake_test2 = test_obj(X={0},S={2,3},Y={1},p_val=0.5)
# fake_test_sentence2 = Sentence(str(fake_test2.to_list()))
# # test_assumptions.add(fake_test_sentence)
# # test_to_sentence_map[fake_test] = fake_test_sentence
# rule = Rule(set([fake_test_sentence2]), fake_test_sentence.contrary())
# ikb_rules.append(rule)
##================================================================

for premise, conclusions in cg.decisions.items():
    p_out_list = []
    for p in premise:
        if type(p) == str:
            p_out = Sentence(p)
            p_out_list.append(p_out)
        elif 'test_obj' in str(p.__class__):
            if p not in test_to_sentence_map:
                test_to_sentence_map[p] = Sentence(str(p.to_list())) 
                IKB_axioms.append(p)
            p_out = test_to_sentence_map[p]
            p_out_list.append(p_out)
        else:
            raise TypeError("Premise is not str or test_obj")
    premise = set(p_out_list)

    for conclusion in conclusions:
        if type(conclusion) == str:
            rule = Rule(premise, Sentence(conclusion))    
            decision_rules.append(rule)
        elif 'test_obj' in str(conclusion.__class__):
            if conclusion not in test_to_sentence_map:
                test_to_sentence_map[conclusion] = Sentence(str(conclusion.to_list()))  
                IKB_axioms.append(conclusion)
            rule = Rule(premise, test_to_sentence_map[conclusion])
            decision_rules.append(rule)
        elif type(conclusion) == tuple:
            for c in conclusion:
                if type(c) == str:
                    rule = Rule(premise, Sentence(c))    
                    decision_rules.append(rule)
                elif 'test_obj' in str(c.__class__):
                    if c not in test_to_sentence_map:
                        test_to_sentence_map[c] = Sentence(str(c.to_list()))  
                        IKB_axioms.append(c)
                    rule = Rule(premise, test_to_sentence_map[c])
                    decision_rules.append(rule)
        else:
            raise TypeError("Conclusion is not str, test_obj or tuple")   

for (a,b) in tqdm(itertools.combinations(IKB_axioms, 2)):

    ### set preference for assumptions based on p-value
    if {test_to_sentence_map[a]}.union({test_to_sentence_map[b]}).issubset(test_assumptions):
        if a.p_val != None and b.p_val != None: ### NOTE: should an inner strength be here already?
            if a.p_val < b.p_val:
                pref = Preference(test_to_sentence_map[a], test_to_sentence_map[b], LESS_THAN)
            elif a.p_val == b.p_val:
                pref = Preference(test_to_sentence_map[a], test_to_sentence_map[b], LESS_EQUAL)
            else:
                pref = Preference(test_to_sentence_map[b], test_to_sentence_map[a], LESS_THAN)
            p_preferences.append(pref)

    ### Contraction
    add = contract(a,b)
    if add != None and add.elements() not in [test.elements() for test in IKB_axioms]:
        IKB_axioms.append(add)
        add_sentence = Sentence(str(add.to_list()))
        test_to_sentence_map[add] = add_sentence
        rule = Rule(set([test_to_sentence_map[a], test_to_sentence_map[b]]), add_sentence)
        contraction_rules.append(rule)
        symm = add.symmetrise()
        if symm != None and symm.elements() not in [test.elements() for test in IKB_axioms]:
            IKB_axioms.append(symm)
            symm_sentence = Sentence(str(symm.to_list()))
            test_to_sentence_map[symm] = symm_sentence
            rule = Rule(set([add_sentence]), symm_sentence)
            symmetry_rules.append(rule)
        ## Decomposition
        decomposed = add.decompose()
        if [a.elements() not in [test.elements() for test in IKB_axioms] for a in decomposed]:
            IKB_axioms=IKB_axioms+decomposed
        decomposed_0_sentence = Sentence(str(decomposed[0].to_list()))
        test_to_sentence_map[decomposed[0]] = decomposed_0_sentence
        rule = Rule(set([add_sentence]), decomposed_0_sentence)
        decomposition_rules.append(rule)
        symm = decomposed[0].symmetrise()
        if symm != None and symm.elements() not in [test.elements() for test in IKB_axioms]:
            IKB_axioms.append(symm)
            symm_sentence = Sentence(str(symm.to_list()))
            test_to_sentence_map[symm] = symm_sentence
            rule = Rule(set([decomposed_0_sentence]), symm_sentence)
            symmetry_rules.append(rule)
        decomposed_1_sentence = Sentence(str(decomposed[1].to_list()))
        test_to_sentence_map[decomposed[1]] = decomposed_1_sentence
        rule = Rule(set([add_sentence]), decomposed_1_sentence)
        decomposition_rules.append(rule)
        symm = decomposed[1].symmetrise()
        if symm != None and symm.elements() not in [test.elements() for test in IKB_axioms]:
            IKB_axioms.append(symm)
            symm_sentence = Sentence(str(symm.to_list()))
            test_to_sentence_map[symm] = symm_sentence
            rule = Rule(set([decomposed_1_sentence]), symm_sentence)
            symmetry_rules.append(rule)
        ## Weak Union
        united = add.weak_union()
        if [a.elements() not in [test.elements() for test in IKB_axioms] for a in united]:
            IKB_axioms=IKB_axioms+united
        united_sentence = Sentence(str(united[0].to_list()))
        test_to_sentence_map[united[0]] = united_sentence
        rule = Rule(set([add_sentence]), united_sentence)
        union_rules.append(rule)
        symm = united[0].symmetrise()
        if symm != None and symm.elements() not in [test.elements() for test in IKB_axioms]:
            IKB_axioms.append(symm)
            symm_sentence = Sentence(str(symm.to_list()))
            test_to_sentence_map[symm] = symm_sentence
            rule = Rule(set([united_sentence]), symm_sentence)
            symmetry_rules.append(rule)
    ### Intersection  
    add = intersect(a,b)
    if add != None and add.elements() not in [test.elements() for test in IKB_axioms]:
        IKB_axioms.append(add)
        add_sentence = Sentence(str(add.to_list()))
        test_to_sentence_map[add] = add_sentence
        rule = Rule(set([test_to_sentence_map[a], test_to_sentence_map[b]]), add_sentence)
        intersection_rules.append(rule)
        symm = add.symmetrise()
        if symm != None and symm.elements() not in [test.elements() for test in IKB_axioms]:
            IKB_axioms.append(symm)
            symm_sentence = Sentence(str(symm.to_list()))
            test_to_sentence_map[symm] = symm_sentence
            rule = Rule(set([add_sentence]), symm_sentence)
            symmetry_rules.append(rule)
        ## Decomposition
        decomposed = add.decompose()
        if [a.elements() not in [test.elements() for test in IKB_axioms] for a in decomposed]:
            IKB_axioms=IKB_axioms+decomposed
        decomposed_0_sentence = Sentence(str(decomposed[0].to_list()))
        test_to_sentence_map[decomposed[0]] = decomposed_0_sentence
        rule = Rule(set([add_sentence]), decomposed_0_sentence)
        decomposition_rules.append(rule)
        symm = decomposed[0].symmetrise()
        if symm != None and symm.elements() not in [test.elements() for test in IKB_axioms]:
            IKB_axioms.append(symm)
            symm_sentence = Sentence(str(symm.to_list()))
            test_to_sentence_map[symm] = symm_sentence
            rule = Rule(set([decomposed_0_sentence]), symm_sentence)
            symmetry_rules.append(rule)
        decomposed_1_sentence = Sentence(str(decomposed[1].to_list()))
        test_to_sentence_map[decomposed[1]] = decomposed_1_sentence
        rule = Rule(set([add_sentence]), decomposed_1_sentence)
        decomposition_rules.append(rule)
        symm = decomposed[1].symmetrise()
        if symm != None and symm.elements() not in [test.elements() for test in IKB_axioms]:
            IKB_axioms.append(symm)
            symm_sentence = Sentence(str(symm.to_list()))
            test_to_sentence_map[symm] = symm_sentence
            rule = Rule(set([decomposed_1_sentence]), symm_sentence)
            symmetry_rules.append(rule)
        ## Weak Union
        united = add.weak_union()
        if [a.elements() not in [test.elements() for test in IKB_axioms] for a in united]:
            IKB_axioms=IKB_axioms+united
        united_sentence = Sentence(str(united[0].to_list()))
        test_to_sentence_map[united[0]] = united_sentence
        rule = Rule(set([add_sentence]), united_sentence)
        union_rules.append(rule)
        symm = united[0].symmetrise()
        if symm != None and symm.elements() not in [test.elements() for test in IKB_axioms]:
            IKB_axioms.append(symm)
            symm_sentence = Sentence(str(symm.to_list()))
            test_to_sentence_map[symm] = symm_sentence
            rule = Rule(set([united_sentence]), symm_sentence)
            symmetry_rules.append(rule)
#======================================================================================#

    ### Define Contraries based on dep_type (p_val<alpha) regardless of S
    # if (a.X==b.X and a.Y==b.Y and a.dep_type != b.dep_type) or (a.X==b.Y and a.Y==b.X and a.dep_type != b.dep_type):
    ### Define Contraries based on S
    # if (a.X==b.X and a.Y==b.Y and a.S != b.S and a.dep_type == b.dep_type == "I") or \
    #     (a.X==b.Y and a.Y==b.X and a.S != b.S and a.dep_type == b.dep_type == "I"):
    #     rule = Rule(set([test_to_sentence_map[a]]), test_to_sentence_map[b].contrary())
    #     contrary_rules.append(rule) 
    #     rule = Rule(set([test_to_sentence_map[b]]), test_to_sentence_map[a].contrary())
    #     contrary_rules.append(rule)
    ### Define Contraries based on dep_type (p_val<alpha)
    elif (a.X==b.X and a.Y==b.Y and a.S == b.S and a.dep_type != b.dep_type ) or \
        (a.X==b.Y and a.Y==b.X and a.S == b.S and a.dep_type != b.dep_type )  :
        rule = Rule(set([test_to_sentence_map[a]]), test_to_sentence_map[b].contrary())
        contrary_rules.append(rule) 
        rule = Rule(set([test_to_sentence_map[b]]), test_to_sentence_map[a].contrary())
        contrary_rules.append(rule)    


ikb_rules = set(contraction_rules+intersection_rules+decomposition_rules+union_rules+symmetry_rules+contrary_rules+decision_rules)
p_preferences = set(p_preferences)

print("Total tests added:",len(IKB_axioms)-len(cg.IKB_list))

print("Number of Contractions applied:",len(contraction_rules))
print("Number of Intersections applied:",len(intersection_rules))
print("Number of Decomposition applied:",len(decomposition_rules))
print("Number of Union applied:",len(union_rules))
print("Number of Symmetry applied:",len(symmetry_rules))
print("Number of Contrary applied:",len(contrary_rules))

print("Number of total rules:", len(ikb_rules))
print("Number of Preferences:", len(p_preferences))
print("Number of LESS_THAN Preferences:", len([pref.relation for pref in p_preferences if pref.relation==1]))
print("Number of LESS_EQUAL Preferences:", len([pref.relation for pref in p_preferences if pref.relation==2]))

print("------------------- Assumptions -------------------")
[print(format_sentence(asm), "p_val=",[round(k.p_val,2) for k, v in test_to_sentence_map.items() if v == asm][0]) for asm in test_assumptions ]

print("------------------- Grounded Rule Set -------------------")
[print_rule(rule) for rule in ikb_rules]

#======================================================================================#
#                                       Build ABA+                                     #
#======================================================================================#

abap = ABA_Plus(assumptions=test_assumptions, rules=ikb_rules, preferences=p_preferences)
# abap.generate_all_deductions(test_assumptions)
deductions = abap.generate_all_deductions(test_assumptions)
[a for a in deductions if a not in test_assumptions]

# args_and_attk = abap.generate_arguments_and_attacks(deductions) ##TODO: Check that it is actually deductions that should feed this
args_and_attk = abap.generate_arguments_and_attacks_for_contraries()

print("Number of Deductions:",len(args_and_attk[0]))

print("Number of Attacks:",len(args_and_attk[1]))
print("Number of Normal Attacks:", len([atk for atk in list(args_and_attk[1]) if atk.type == NORMAL_ATK]))
print("Number of Reverse Attacks:", len([atk for atk in list(args_and_attk[1]) if atk.type == REVERSE_ATK]))

print("Number of All Deductions:",len(args_and_attk[2]))

print("------------------- Deductions -------------------")
# There can be multiple deductions per assumption
[print(format_deduction_set(ded)) for ded in list(args_and_attk[0].values()) ] #if len(ded)>1
[print(deduce) for deduce in [format_deduction_set(ded) for ded in list(args_and_attk[0].values())] if "orient" in deduce]

print("------------------- Attacks -------------------")
[print_attack(atk) for atk in list(args_and_attk[1])]
print("------------------- All Deductions -------------------")
# Set of all deductions obtained 
[print(format_deduction(ded)) for ded in list(args_and_attk[2]) ] #if len(ded.premise)>1
[print(deduce) for deduce in [format_deduction(ded) for ded in list(args_and_attk[2])] if "orient" in deduce]


## Dump the results in a file
items = [format_deduction_set(ded) for ded in args_and_attk[0].values()]
with open('Deductions.txt', 'w') as f:
    f.write('\n'.join(items))
f.close()

items = [format_attack(atk) for atk in list(args_and_attk[1])]
with open('Attacks.txt', 'w') as f:
    f.write('\n'.join(items))
f.close()

items = [format_deduction(ded) for ded in list(args_and_attk[2])]
with open('AllDeductions.txt', 'w') as f:
    f.write('\n'.join(items))
f.close()
#------------------------------  Calculate Extensions  -------------------------------#
from ABAplus.aspartix_interface import ASPARTIX_Interface
asp = ASPARTIX_Interface(abap)
asp.generate_input_file_for_clingo("test.lp")
stable = asp.calculate_stable_extensions("test.lp")
complete = asp.calculate_complete_extensions("test.lp")
# admissible = asp.calculate_admissible_extensions("test.lp")
preferred = asp.calculate_preferred_extensions("test.lp")
grounded = asp.calculate_grounded_extensions("test.lp")

[format_sentence(ded) for ded in next(iter(abap.generate_all_deductions(stable)))]
format_sets(stable)

#======================================================================================#
#                         Revise the causal graph with ABA results                     #
#======================================================================================#

## Check if current orientations are in the accepted extension

## Check what D-separation rules apply given the accepted tests

##NOTE: do we need both?

### Can start by checking the rejected assumptions




#======================================================================================#
#                           Code Gradual Semantics (T-Norms)                           #
#======================================================================================#



#======================================================================================#
#                                   Checks and Debugging                               #
#======================================================================================#
### Debugging
## within generate_arguments_and_attacks() function
## List of all deductions
# [a for ded in deductions.values() for a in list(ded) ]
## Maximum deduction per assumption
# max([len(list(ded)) for ded in deductions.values()])
## Stable extension results
# next(iter(next(iter(stable)))) in args_and_attk[0]
# next(iter(stable)).intersection(args_and_attk[0])

accepted_indep = [s.symbol for s in next(iter(stable)) if 'I' in s.symbol]

### Check if extension is the same as the set of assumptions
stable_print = [s.symbol for s in next(iter(stable)) ]
ass_print = [s.symbol for s in test_assumptions]
set(stable_print) == set(ass_print)

abap.generate_all_deductions(decision_rules)

##Checks
[test.to_list() for test in cg.IKB_list if test.p_val == None]
[test.to_list() for test in IKB_axioms if test.p_val == None]


def print_relevant_tests_from_list(test_list:list, x:set, y:set, S:set=None, p_val:str=None, d_type:str=None)->list:
    if S != None: ## s==set() for tests with empty set - not {}
        if not d_type:
            return [test.to_list() for test in test_list if test.X==x and test.Y==y and test.S==S and test.p_val == p_val]
        else:
            return [test.to_list() for test in test_list if test.X==x and test.Y==y and test.S==S and test.p_val == p_val and test.dep_type==d_type]
    else:
        if not d_type:
            return [test.to_list() for test in test_list if test.X==x and test.Y==y and test.p_val == p_val]
        else:
            return [test.to_list() for test in test_list if test.X==x and test.Y==y and test.p_val == p_val and test.dep_type==d_type]

print_relevant_tests_from_list(test_list=cg.IKB_list, x={0}, y={1}, S=set(), p_val='all', d_type=None)

