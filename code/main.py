import argparse
import random
import logging
import json
import pdb
import numpy as np
from src.base import YAGO, DBPedia, Freebase, MetaQA
from src.user import query_log_by_mids, query_log_by_topics
from src.apex import APEX, APEX_N

logging.basicConfig(format='[%(asctime)s] - %(message)s',
                    level=logging.DEBUG,
                    filename = 'log.log', filemode= 'a')

class SummaryMethod(object):
    """Stores a summarization function and associated metadata."""

    def __init__(self, fn, name, **kwargs):
        """
        :param fn: summarization function to call
        :param name: pretty-printed name of this function
        :param kwargs: optional keyword arguments for fn
        """
        self.fn_ = fn
        self.name_ = name
        self.kwargs_ = kwargs

    def name(self):
        return self.name_

    def kwargs(self):
        return self.kwargs_

    def __call__(self, KG, K, query_log):
        """
        :param KG: KnowledgeGraph
        :param K: summary constraint
        :param query_log: query log
        :return results: results of function call
        """
        return self.fn_(KG, K, query_log, **self.kwargs_)
    

# Available choices for user input arguments in main
# TODO: Change these to point to your local data directories
KG_MAPPING = {
    'YAGO': YAGO(rdf_gz='yagoFacts.gz', query_dir='final/', mid_dir='by-mid/'),
    'Freebase': Freebase(query_dir='queries/final/'),
    'DBpedia': DBPedia(),
    'MetaQA': MetaQA(),
}

METHODS = { 
    'apex': SummaryMethod(APEX, 'APEX'),
    'apex-n': SummaryMethod(APEX_N, 'APEX-N'),
}

def answer_queries_in_log(KG, K, query_log, summary_methods, acc_list_apex_n, acc_list_apex, update_time_apex_n, update_time_apex, query_num_per_test=1, gamma=0.5):
    """
    :param KG: KnowledgeGraph
    :param K: summary constraint
    :param query_log: list of dict
    :param summary_methods: summarization methods to use
    :param test_size: percent of queries to hold out for testing
    """

    for summary_method in summary_methods:
        logging.info('\t---Summarizing with {}---'.format(summary_method.name()))

        # APEX
        if summary_method.name_ == 'APEX-N':
            acc_list, update_time_list = APEX_N(KG, K, query_log, query_num_per_test, gamma=gamma)
            acc_list_apex_n += acc_list
            update_time_apex_n += update_time_list
        if summary_method.name_ == 'APEX': 
            acc_list, update_time_list = APEX(KG, K, query_log, query_num_per_test, gamma=gamma)
            acc_list_apex += acc_list
            update_time_apex += update_time_list


def float_in_zero_one(value):
    """Check if a float value is in [0, 1]"""
    value = float(value)
    if value < 0 or value > 1:
        raise argparse.ArgumentTypeError('Value must be a float between 0 and 1')
    return value

def positive_int(value):
    """Check if an integer value is positive"""
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('Value must be positive')
    return value

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--kg', choices=list(KG_MAPPING.keys()), default='MetaQA',
            help='KG to summarize')
    parser.add_argument('--n-queries', type=positive_int, default=200,
            help='Number of queries to simulate per user. Default is 200.')
    parser.add_argument('--n-topic-mids', type=positive_int, default=20,
            help='Number of topic mids of interest per user. Default is 20.')
    parser.add_argument('--n-topics', type=positive_int, default=10,
            help='Number of topics to simulate per user log. '
                 'For Freebase only. Default is 10.')
    parser.add_argument('--n_users', type=positive_int, default=2,
            help='Number of users to simulate. Default is 2.')
    parser.add_argument('--start_user', type=positive_int, default=0,
            help='Number of users to simulate. Default is 0.')
    parser.add_argument('--percent-triples', type=float_in_zero_one, default=0.0001,
            help='Ratio of number of triples of KG to use as K '
                 '(summary constraint). Default is 0.001.')
    parser.add_argument('--random-query-prob', type=float_in_zero_one, default=0,
            help='Probability of users asking random queries rather '
                 'than topic-specific ones. Default is 0.')
    parser.add_argument('--shuffle', action='store_true',
            help='Set this flag to true to shuffle all generated logs. Default False.')
    parser.add_argument('--method', nargs='+', default=['apex'],
            choices=list(METHODS.keys()),
            help='Summarization methods to call. Default is [apex].')
    parser.add_argument('--save-queries', action='store_true', 
            help='Whether to save the generated queries (warning: overwrite existing saves). Default False')
    parser.add_argument('--load-queries', action='store_true', 
            help='Whether to load queries from dataset folder (warning: make sure dataset folder is non-empty, also make sure enough amount of queries than acquired). Default False and default is to generate new queries')
    parser.add_argument('--query-num-per-test', type=positive_int, default=1,
            help='Number of users to simulate. Default is 1.')
    parser.add_argument('--gamma', type=float_in_zero_one, default=0.5,
            help='Decay factor for APEX and APEX-N. Default is 0.5.')
    
    return parser.parse_args()

def main():
    acc_list_apex_n = []
    acc_list_apex = []
    update_time_apex_n = []
    update_time_apex = []
    args = parse_args()
    if (args.save_queries is True) and (args.load_queries is True):
        raise NotImplementedError

    if (args.save_queries is True):
        print('Warning: File Rewrite, proceed?')
        pdb.set_trace()


    KG = KG_MAPPING[args.kg]
    summary_methods = [METHODS[name] for name in args.method]

    # Load the KG into memory
    logging.info('Loading {}'.format(KG.name()))
    KG.load()
    logging.info('Loaded {}'.format(KG.name()))


    # Number of triples for summary
    K = int(args.percent_triples * KG.number_of_triples())
    logging.info('K = {}'.format(K))

    # Simulate users with specified parameters
    for user in range(args.start_user, args.start_user + args.n_users):
        logging.info('---Simulating user {}---'.format(user))
        KG.update_user(user)
        logging.basicConfig(filename=KG.user_dir + '/log.log')

        if args.kg == 'Freebase':
            topics = random.sample(KG.topics(), k=args.n_topics)

            query_log = query_log_by_topics(
                    KG, topics, args.n_mids_per_topic, args.n_queries,
                    shuffle=args.shuffle, random_query_prob=args.random_query_prob)
        
        elif args.kg == 'MetaQA':
            args.save_queries = False
            args.load_queries = True
            query_log = []
            for i in range(args.n_queries):
                with open(KG.query_dir() + "q"+ str(i)+ ".json", "r") as f:
                    query_log.append(json.load(f))

            logging.info('---Loaded a log of {} queries----'.format(len(query_log)))
        

        else:
            # topic_mids = random.sample(KG.topic_mids(), k=args.n_topic_mids)
            topic_mids = random.sample(list(KG.triples_.keys()), k=args.n_topic_mids)

            if args.load_queries:
                query_log = []
                for i in range(args.n_queries):
                    with open(KG.query_dir() + "q"+ str(i)+ ".json", "r") as f:
                        query_log.append(json.load(f))

                logging.info('---Loaded a log of {} queries----'.format(len(query_log)))

            else:

                query_log = query_log_by_mids(
                        KG, topic_mids, args.n_queries, topic_dist=np.ones(len(topic_mids))/len(topic_mids),
                        shuffle=args.shuffle,
                        random_query_prob=args.random_query_prob, whether_save = args.save_queries)
                logging.info('---Generated a log of {} queries----'.format(len(query_log)))

        if args.save_queries:
            logging.info('---Saving generated queries---')
            for i in range(len(query_log)):
                with open(KG.query_dir() + "q"+ str(i)+ ".json", "w") as outfile:
                    json.dump(query_log[i], outfile)
        answer_queries_in_log(KG, K, query_log, summary_methods, acc_list_apex_n, acc_list_apex, update_time_apex_n, update_time_apex, query_num_per_test=args.query_num_per_test, gamma=args.gamma)

    # save arrays
    if len(acc_list_apex) > 0:
        print('Average F1 score for APEX: ', np.mean(acc_list_apex))
    if len(update_time_apex) > 0:
        print('Average update time for APEX (seconds): ', np.mean(update_time_apex))
    if len(acc_list_apex_n) > 0:
        print('Average F1 score for APEX-N: ', np.mean(acc_list_apex_n))
    if len(update_time_apex_n) > 0:
        print('Average update time for APEX-N (seconds): ', np.mean(update_time_apex_n))

    logging.info('The program has finished running. For more information, please check the log file')
if __name__ == '__main__':
    main()


