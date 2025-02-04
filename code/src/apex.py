import numpy as np
import scipy.sparse as sp
from .base import KnowledgeGraph
from time import time
from .metrics import total_query_log_metrics, average_query_log_metrics
import logging
import random
from tqdm import tqdm
DETAILED_LOGGING = True
GAMMA = 0.5

class Summary(KnowledgeGraph):

    def __init__(self, KG):
        """
        :param KG: KnowledgeGraph
        """
        super().__init__()
        self.parent_ = KG

    def parent(self):
        return self.parent_

    def fill(self, triples, k):
        """
        :param triples: triples to add to summary
        :param k: limit
        """
        for triple in triples:
            if self.number_of_triples() >= k:
                return
            self.add_triple(triple)

    def add_entity(self, topic_mid):
        for entity in self.entities().copy():
            if self.parent().csr_matrix()[self.parent().entity_id_[entity], self.parent().entity_id_[topic_mid]] != 0: # exists a triple
                # find the relationship
                for r in self.parent().entities_to_relation_[entity][topic_mid]:
                    self.add_triple((entity, r, topic_mid))

            if self.parent().csr_matrix_T()[self.parent().entity_id_[entity], self.parent().entity_id_[topic_mid]] != 0: # exists a triple
                for r in self.parent().entities_to_relation_[topic_mid][entity]:
                    self.add_triple((topic_mid, r, entity))

        # in case self.entities() is empty
        self.entities_.add(topic_mid)

    
    def add_random_triple(self):
        e1 = random.choice(list(self.parent().triples_.keys()))
        r = random.choice(list(self.parent()[e1].keys()))
        e2 = random.choice(list(self.parent()[e1][r]))
        self.add_triple((e1, r, e2))


def Heat_Diffuse(heat, KG, query, d, csr_indirect_matrices): # adj matrix and heat
    q = np.zeros(KG.number_of_entities())
    parse = query['Parse']
    topic_mid = parse['TopicEntityMid']
    topic_eid = KG.entity_id(parse['TopicEntityMid'])
    relation = parse['InferentialChain'][0]
    answers = KG[topic_mid][relation]
    q[topic_eid] += 1
    answers_rid = np.zeros(len(answers), dtype = int)
    for i, item in enumerate(answers):
        answers_rid[i] = KG.entity_id(item)
        q[answers_rid[i]] += 1/len(answers)


    # diffuse through edges
    r = q.copy()
    for i in range(d):
        q = csr_indirect_matrices[i+1] * q
        r += q
    changed_index_list = []
    for i in range(len(r)):
        if r[i] != 0 and r[i] > 1e-3:
            changed_index_list.append(i)
        else:
            r[i] = 0
    r += heat

    return r, changed_index_list


def heat_decay_array(h, gamma, threshold = 1e-3):
    h = h * gamma
    for i in range(len(h)):
        if h[i] < threshold:
            h[i] = 0


def binary_find(index_list, changed_index, H):
    changed_value = H[changed_index]
    low = 0
    high = len(index_list) - 1
    mid = 0
    while high >= low:
        mid = (high + low)//2
        if H[index_list[mid]] < changed_value:
            high = mid - 1
        elif H[index_list[mid]] > changed_value:
            low = mid + 1
        else:
            if (index_list[mid] == changed_index):
                return mid
            else:
                mid_right = mid + 1
                mid_left = mid - 1
                while mid_right < len(index_list) and H[index_list[mid_right]] == changed_value:
                    if index_list[mid_right] == changed_index:
                        return mid_right
                    mid_right += 1
                while mid_left >= 0 and H[index_list[mid_left]] == changed_value:
                    if index_list[mid_left] == changed_index:
                        return mid_left
                    mid_left -= 1
    return -1


def binary_insert(index_list, changed_index, H):

    changed_value = H[changed_index]
    low = 0
    high = len(index_list) - 1

    if high == -1: # empty list
        index_list.append(changed_index)
        return
    
    # check side case
    if changed_value >= H[index_list[0]]:
        index_list.insert(0, changed_index)
        return
    if changed_value <= H[index_list[len(index_list) - 1]]:
        index_list.insert(len(index_list), changed_index)
        return 
    
    while high >= low + 2:
        mid = (high + low)//2
        if H[index_list[mid]] == changed_value:
            index_list.insert(mid, changed_index)
            return
        if H[index_list[mid]] < changed_value:
            high = mid
        else:
            low = mid

    #  low <= high <= low + 1
    index_list.insert(high, changed_index)


# not finished yet
def incremental_sort(index_list, changed_index_list, H_old, H_new):   # changed_index_list and prev_values are 1-1 corresponse
    # delete the changed elements
    positions = []
    for index in changed_index_list:
        pos = binary_find(index_list, index, H_old)
        if pos != -1:
            positions.append(pos)
    for index_del in sorted(positions, reverse=True):
        del index_list[index_del]

    # re-insert the changed elements
    for index in changed_index_list:
        if H_new[index] != 0:
            binary_insert(index_list, index, H_new)

    


def construct_naive(KG, K, topic_eids):
    P = Summary(KG)
    topic_mids = []
    # turn eid into mid
    for topic_eid in topic_eids:
        topic_mids.append(KG.id_entity_[topic_eid])
    i = 0
    while P.number_of_triples() < K:
        P.add_entity(topic_mids[i])
        i += 1
        if i == len(topic_eids):
            while P.number_of_triples() < K:
                P.add_random_triple()
            break
    return P

    
def construct_complete(KG, K, index_list):
    P = Summary(KG)
    need_random = 1

    for i in range(len(index_list)):
        triple_to_add = (KG.id_entity(index_list[i][0]), KG.id_relation(index_list[i][1]), KG.id_entity(index_list[i][2]))
        P.add_triple(triple_to_add)
        if P.number_of_triples() >= K:
            need_random = 0
            break

    if need_random:
        need = K - P.number_of_triples()
        triple_to_add = random.sample(KG.triples(), k = need)
        for triple in triple_to_add:
            P.add_triple(triple)
    return P


def APEX_N(KG, K, query_log, query_num_per_test=1, gamma=GAMMA, diameter = 1, alpha=0.3):
    print('Running APEX-N')
    update_time_list = []
    acc_list = []
    t = 0
    num_queries_total = len(query_log)
    csr_indirect_matrices = {}
    csr_indirect_matrices[0] = sp.eye(KG.number_of_entities())
    print('calculating heat matrices (one-time computing)')
    for i in range(diameter):
        csr_indirect_matrices[i+1] = alpha * KG.csr_matrix_indirect_heat() * csr_indirect_matrices[i]
    # initialize heat

    print('heat matrices calculated')
    initial_heat = np.zeros(KG.number_of_entities())
    index_list = []
    first_q = query_log[0]
    heat, _ = Heat_Diffuse(initial_heat, KG, first_q, diameter, csr_indirect_matrices)
    sorted_heat = np.sort(heat)[::-1]
    sorted_args = np.argsort(heat)[::-1]
    i = 0
    while sorted_heat[i] > 0:
        index_list.append(sorted_args[i])
        i += 1
    P = construct_naive(KG, K, topic_eids = index_list)
    # test for initial graph
    if DETAILED_LOGGING:
        logging.info('Initial Test')
    total_F1, total_precision, total_recall = total_query_log_metrics(P, query_log[1: 1+query_num_per_test])
    if DETAILED_LOGGING:
        logging.info('\t  Total F1/precision/recall')
        logging.info('\t    {:.2f}/{:.2f}/{:.2f}'.format(
            total_F1, total_precision, total_recall))
    
    avg_F1, avg_precision, avg_recall = average_query_log_metrics(P, query_log[1: 1+query_num_per_test])
    if DETAILED_LOGGING:
        logging.info('\t  Average F1/precision/recall')
        logging.info('\t    {:.2f}/{:.2f}/{:.2f}'.format(
            avg_F1, avg_precision, avg_recall))

    # update phase
    # while t < num_queries_total - query_num_per_test:
    print('Adaptive personalized knowledge graph summarization for {} timestamps'.format(num_queries_total - query_num_per_test - 1))
    for t in tqdm(range(1, num_queries_total - query_num_per_test)):
        t0 = time()
        heat = heat * gamma
        heat_new, changed_index_list = Heat_Diffuse(heat, KG, query_log[t], diameter, csr_indirect_matrices)
        for i in range(len(heat_new)):
            if heat_new[i] > 0 and heat_new[i] < 1e-4:
                heat_new[i] = 0

        incremental_sort(index_list, changed_index_list, heat, heat_new)
        # print('incremental end')
        new_index_list = []
        for i in range(len(index_list)):
            if heat_new[index_list[i]] != 0:
                new_index_list.append(index_list[i])
            else:
                break
        index_list = new_index_list
        heat = heat_new
        P = construct_naive(KG, K, topic_eids= index_list)
        update_time = time() - t0
        update_time_list.append(update_time)

        if DETAILED_LOGGING:
            logging.info('\t  Adapting for time: {}'.format(t))
            logging.info('\t  Time: {:.2f} seconds'.format(update_time))

        # Evaluate question answering on the testing queries
        total_F1, total_precision, total_recall = total_query_log_metrics(P, query_log[t+1: t+1+query_num_per_test])
        if DETAILED_LOGGING:
            logging.info('\t  Total F1/precision/recall')
            logging.info('\t    {:.2f}/{:.2f}/{:.2f}'.format(
                total_F1, total_precision, total_recall))
        
        avg_F1, avg_precision, avg_recall = average_query_log_metrics(P, query_log[t+1: t+1+query_num_per_test])
        if DETAILED_LOGGING:
            logging.info('\t  Average F1/precision/recall')
            logging.info('\t    {:.2f}/{:.2f}/{:.2f}'.format(
                avg_F1, avg_precision, avg_recall))
        
        acc_list.append(avg_F1)
        t += 1     

    logging.info('\t  Ave Time on Each Training Log: {:.2f} seconds'.format(np.mean(update_time_list)))
    logging.info('\t  Ave Ave F1 on Each Training Log: {:.2f}'.format(np.mean(acc_list)))

    return acc_list, update_time_list



def APEX(KG, K, query_log, query_num_per_test=1, gamma=GAMMA, diameter = 1, alpha=0.3):
    print('Running APEX')
    smoothing = 0.5
    update_time_list = []
    acc_list = []
    t = 0
    num_queries_total = len(query_log)
    csr_indirect_matrices = {}
    csr_indirect_matrices[0] = sp.eye(KG.number_of_entities())
    print('calculating heat matrices (one-time computing)')
    for i in range(diameter):
        csr_indirect_matrices[i+1] = alpha * KG.csr_matrix_indirect_heat() * csr_indirect_matrices[i]

    print('heat matrices calculated')
    # initial state
    index_list = []
    q = np.zeros(KG.number_of_entities())
    e = np.zeros(KG.number_of_entities())
    r = np.zeros(KG.number_of_relationships())

    # q (query), e (entity heat) and r after first query
    first_q = query_log[0]
    parse = first_q['Parse']
    topic_mid = parse['TopicEntityMid']
    topic_eid = KG.entity_id(parse['TopicEntityMid'])
    relation = parse['InferentialChain'][0]
    answers = KG[topic_mid][relation]
    q[topic_eid] += 1
    answers_rid = np.zeros(len(answers), dtype = int)
    for i, item in enumerate(answers):
        answers_rid[i] = KG.entity_id(item)
        q[answers_rid[i]] += 1/len(answers)

    for i in range(diameter):
        e += csr_indirect_matrices[i] * q
    
    r[KG.relation_id(relation)] = 1    

    # the Heat
    H = {}
    for triple in KG.triples():
        e1, re, e2 = triple
        e1 = KG.entity_id(e1)
        re = KG.relation_id(re)
        e2 = KG.entity_id(e2)
        if e[e1] != 0 and r[re] != 0 and e[e2] != 0:
            H[(e1, re, e2)] = e[e1] * (r[re] + smoothing) * e[e2]
            index_list.append((e1, re, e2))
        
    # sort index_list
    h_for_sort = []
    for index in index_list:
        h_for_sort.append(H[index])

    h_for_sort = np.array(h_for_sort)

    sorted_args = np.argsort(h_for_sort)[::-1]
    sorted_index_list = []
    for i in range(len(sorted_args)):
        sorted_index_list.append(index_list[sorted_args[i]])
    index_list = sorted_index_list

    # construct initial summary
    P = construct_complete(KG, K, index_list)
    # test for initial graph
    if DETAILED_LOGGING:
        logging.info('Initial Test')
    total_F1, total_precision, total_recall = total_query_log_metrics(P, query_log[1: 1+query_num_per_test])
    if DETAILED_LOGGING:
        logging.info('\t  Total F1/precision/recall')
        logging.info('\t    {:.2f}/{:.2f}/{:.2f}'.format(
            total_F1, total_precision, total_recall))
    
    avg_F1, avg_precision, avg_recall = average_query_log_metrics(P, query_log[1: 1+query_num_per_test])
    if DETAILED_LOGGING:
        logging.info('\t  Average F1/precision/recall')
        logging.info('\t    {:.2f}/{:.2f}/{:.2f}'.format(
            avg_F1, avg_precision, avg_recall))
        

    # update phase
    t += 1
    # while t < num_queries_total - query_num_per_test:
    print('Adaptive personalized knowledge graph summarization for {} timestamps'.format(num_queries_total - query_num_per_test - 1))
    for t in tqdm(range(1, num_queries_total - query_num_per_test)):
        t0 = time()
        H_new = {}
        for key in H:
            H_new[key] = H[key]*gamma**3
        # incremental update
        q_T = np.zeros(KG.number_of_entities())
        parse = query_log[t]['Parse']
        topic_mid = parse['TopicEntityMid']
        topic_eid = KG.entity_id(parse['TopicEntityMid'])
        relation = parse['InferentialChain'][0]
        answers = KG[topic_mid][relation]
        q_T[topic_eid] += 1
        answers_rid = np.zeros(len(answers), dtype = int)
        for i, item in enumerate(answers):
            answers_rid[i] = KG.entity_id(item)
            q_T[answers_rid[i]] += 1/len(answers)

        q = q*gamma + q_T
        e_new, changed_entity_list = Heat_Diffuse(e, KG, query_log[t], diameter, csr_indirect_matrices)
        r = r*gamma
        r[KG.relation_id(relation)] += 1

        for i in H_new:
            if H_new[i] > 0 and H_new[i] < 1e-5:
                H_new[i] = 0
        
        changed_triples_in_id = set()


        for changed_entity_id in changed_entity_list:
            potential_triples_vector = KG.csr_matrix_indirect()[changed_entity_id, :]
            rows, cols = potential_triples_vector.nonzero()
            for another_entity_id in cols:
                if e[another_entity_id] == 0 and e_new[another_entity_id] == 0:
                    continue
                triple_found = KG.find_triple_by_entities_indirect(changed_entity_id, another_entity_id)
                if triple_found is not None:
                    # for triple_ in triple_found:
                    if triple_found[1] < KG.number_of_relationships():
                        if triple_found[1] != relation and r[triple_found[1]] == 0:
                            continue
                        changed_triples_in_id.add(triple_found)

        e = e_new

        for triple in KG.triples_by_relation_id(KG.relation_id(relation)):
            changed_triples_in_id.add(triple)

        changed_triples_in_id_list = list(changed_triples_in_id)
        for triple_in_id in changed_triples_in_id_list:
            new_value = e[triple_in_id[0]] * ((r[triple_in_id[1]]) + smoothing) * e[triple_in_id[2]]
            if new_value > 0:
                H_new[triple_in_id] = new_value
            else:
                changed_triples_in_id.remove(triple_in_id)


        for triple_in_id in list(changed_triples_in_id):
            if triple_in_id not in H:
                H[triple_in_id] = 0


        # incremental sort
        incremental_sort(index_list, changed_triples_in_id, H, H_new)

        new_index_list = []
        for i in range(len(index_list)):
            if H_new[index_list[i]] != 0:
                new_index_list.append(index_list[i])
            else:
                break
        index_list = new_index_list

        H = H_new

        P = construct_complete(KG, K, index_list)
        update_time = time() - t0
        update_time_list.append(update_time)

        if DETAILED_LOGGING:
            logging.info('\t  Adapting for time: {}'.format(t))
            logging.info('\t  Time: {:.2f} seconds'.format(update_time))

        # Evaluate question answering on the testing queries
        total_F1, total_precision, total_recall = total_query_log_metrics(P, query_log[t+1: t+1+query_num_per_test])
        if DETAILED_LOGGING:
            logging.info('\t  Total F1/precision/recall')
            logging.info('\t    {:.2f}/{:.2f}/{:.2f}'.format(
                total_F1, total_precision, total_recall))
        
        avg_F1, avg_precision, avg_recall = average_query_log_metrics(P, query_log[t+1: t+1+query_num_per_test])
        if DETAILED_LOGGING:
            logging.info('\t  Average F1/precision/recall')
            logging.info('\t    {:.2f}/{:.2f}/{:.2f}'.format(
                avg_F1, avg_precision, avg_recall))

        acc_list.append(avg_F1)
        t += 1

    logging.info('\t  Ave Time on Each Training Log: {:.2f} seconds'.format(np.mean(update_time_list)))
    logging.info('\t  Ave Ave F1 on Each Training Log: {:.2f}'.format(np.mean(acc_list)))

    return acc_list, update_time_list


