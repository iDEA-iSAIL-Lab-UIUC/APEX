import numpy as np
import random
import argparse
import dataprocessing
import pdb
import os
import json

def positive_int(value):
    """Check if an integer value is positive"""
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('Value must be positive')
    return value

def float_in_zero_one(value):
    """Check if a float value is in [0, 1]"""
    value = float(value)
    if value < 0 or value > 1:
        raise argparse.ArgumentTypeError('Value must be a float between 0 and 1')
    return value

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-queries', type=positive_int, default=200,
            help='Number of queries to simulate per user. Default is 200.')
    parser.add_argument('--n-topic-mids', type=positive_int, default=20,
            help='Number of topic mids of interest per user. Default is 20.')
    parser.add_argument('--n_users', type=positive_int, default=10,
            help='Number of users to simulate. Default is 10.')
    parser.add_argument('--random-query-prob', type=float_in_zero_one, default=0,
            help='Probability of users asking random queries rather '
                 'than topic-specific ones. Default is 0.')
    parser.add_argument('--shuffle', action='store_true',
            help='Set this flag to true to shuffle all generated logs. Default False.')
    
    return parser.parse_args()

def generate_query(topic_mid, n_mid_queries, qid):
    inferential_chain = []
    constraints = []
    query_triple = random.sample(dataprocessing.entity_to_query[topic_mid], k = 1)[0]
    inferential_chain.append(query_triple[1])
    answers = list(query_triple[2])
    return {
        'QuestionId': str(qid),
        'Parse': {
            'TopicEntityMid': topic_mid,
            'TopicEntityName': topic_mid,
            'InferentialChain': inferential_chain,
            'Constraints': constraints,
            'Answers': [ {
                    'AnswerType': 'Entity',
                    'AnswerArgument': answer,
                    'EntityName': answer
                } for answer in answers
            ]
        }
    }
    

def generate_queries_by_mid(topic_mid, n_mid_queries, start_qid):
    a = [generate_query(topic_mid, n_mid_queries, qid = _ + start_qid) for _ in range(n_mid_queries)]
    return a

if __name__ == '__main__':
    args = parse_args()

    for i in range(args.n_users):
        start_qid = 0
        query_log = []
        topic_mids = random.sample(list(dataprocessing.entity_to_query.keys()), k = args.n_topic_mids)
        n_topics = len(topic_mids)
        # topic_dist = np.random.uniform(size = n_topics)
        # topic_dist /= np.sum(topic_dist)
        topic_dist = np.ones(len(topic_mids))/len(topic_mids)
        queries_per_topic = np.int64(np.ceil(topic_dist * args.n_queries))
        for n_mid_queries, topic_mid in zip(queries_per_topic, topic_mids):
                n_mid_queries = min(n_mid_queries, args.n_queries - len(query_log))
                query_log.extend(generate_queries_by_mid(
                    topic_mid, n_mid_queries, start_qid))
                start_qid += n_mid_queries

        # save the log
        USER_DIR = './queries' + '/user' + str(i)
        mid_dir_ = USER_DIR + '/by-mid/'
        query_dir_ = USER_DIR + '/final/'

        CHECK_FOLDER = os.path.isdir(USER_DIR)
        # If folder doesn't exist, then create it.
        if not CHECK_FOLDER:
            os.makedirs(USER_DIR)

        CHECK_QUERY_FOLDER = os.path.isdir(query_dir_)
        if not CHECK_QUERY_FOLDER:
            os.makedirs(query_dir_)

        CHECK_MID_FOLDER = os.path.isdir(mid_dir_)
        if not CHECK_MID_FOLDER:
            os.makedirs(mid_dir_)

        print('---Clear existing mid and query files---')
        for filename in os.listdir(mid_dir_):
            file_path = os.path.join(mid_dir_, filename)
            os.unlink(file_path)
        for filename in os.listdir(query_dir_):
            file_path = os.path.join(query_dir_, filename)
            os.unlink(file_path)

        print('---Saving chosen mids---')
        indices = []
        for i in range(len(topic_mids)):
            with open(mid_dir_ + topic_mids[i] + ".list", "w") as mid_outfile:
                for j in range(queries_per_topic[i]):
                    if j + np.sum(queries_per_topic[:i]) not in indices:
                        mid_outfile.write('q' + str(j + np.sum(queries_per_topic[:i])) + '\n') 

        print('---Saving final queries---')
        for i in range(len(query_log)):
                with open(query_dir_ + "q"+ str(i)+ ".json", "w") as outfile:
                    json.dump(query_log[i], outfile)
    

        print('---Program finished---')