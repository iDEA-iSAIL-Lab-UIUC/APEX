import json
import pdb

relation_translate = {'has_tags': ['movie_to_tags', 'tag_to_movie'], 'starred_actors': ['movie_to_actor', 'actor_to_movie'], 'directed_by': ['movie_to_director', 'director_to_movie'], 'written_by': ['movie_to_writer', 'writer_to_movie'], 'release_year': ['movie_to_year'], 'has_imdb_votes': ['movie_to_imdbvotes'], 
                      'has_imdb_rating': ['movie_to_imdbrating'], 'in_language': ['movie_to_language'], 'has_genre': ['movie_to_genre']}

all_entities = set()

triples = {}
triples_find_relation = {}
with open('kb.txt', encoding='utf-8') as f:
    for line in f:
        fact = tuple(line.rstrip().split('|'))
        e1, r, e2 = fact

        all_entities.add(e1)
        all_entities.add(e2)

        if e1 not in triples:
            triples[e1] = {}
        if r not in triples[e1]:
            triples[e1][r] = set()
        triples[e1][r].add(e2)

        if e1 not in triples_find_relation:
            triples_find_relation[e1] = {}
        if e2 not in triples_find_relation[e1]:
            triples_find_relation[e1][e2] = set()
        triples_find_relation[e1][e2].add(r)

entity_to_query = {}
    
relation_file = open('./1-hop/qa_train_qtype.txt', 'r', encoding='utf-8')
relations = relation_file.readlines()
relation_file.close()

queries = {}
query_id = 0
with open('./1-hop/vanilla/qa_train.txt', encoding='utf-8') as f:
    for line in f:
        q, a = tuple(line.rstrip().split('\t'))
        query_relation = relations[query_id].rstrip()
        ind1 = 0
        ind2 = 0
        for i in range(len(q)):
            if q[i] == '[':
                ind1 = i
            if q[i] == ']':
                ind2 = i
        query_entity = q[ind1 + 1: ind2]
        answers = a.split('|')
        # for answer in answers:
        if query_entity not in entity_to_query:
            entity_to_query[query_entity] = set()
        entity_to_query[query_entity].add((query_entity, query_relation, tuple(answers)))
        query_id += 1
