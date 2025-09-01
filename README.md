# APEX<sup>2</sup>: Adaptive and Extreme Summarization for Personalized Knowledge Graphs
![Overview](Apex-Legends.jpg)
Figure credit: (one of) my favorite video game [APEX Legends](https://en.wikipedia.org/wiki/Apex_Legends). This project is dedicated to the memory of my undergraduate years.

------------------------
**[Paper (full arXiv version)](./KDD2025_APEX2_arXiv.pdf)** | **[Paper (ACM version)](https://dl.acm.org/doi/abs/10.1145/3690624.3709213)** | **[Poster](./KDD_2025_Poster.pdf)** | **[Video Summary](https://www.youtube.com/watch?v=FtcNqk7rX40)**

## file structure
    - README.md (this file)
    - .gitignore
    - code
        - src (utils)
        - main.py (run this, specific commands will be provided below)
        
    - PKG_EXP
        - DBpedia
            - queries (query folder, empty, will be generated)
                - user # (user folder)
            - facts.gz (dataset, download instructions below)
        - MetaQA    
            kb.txt (dataset)
            - dataprocessing.py (utils when generating queries)
            - generate_query.py (can be used to re-generate queries)
            - queries (query folder)
                - user # (user folder)
        - YAGO
            - queries (query folder, empty, will be generated)
                - user # (user folder)
            - yagoFacts.gz, missing large dataset file, need to be manually downloaded. Instructions in both paper and below, 


## Datasets
We provide MetaQA that we used in our experiments along with the source code. 

DBpedia can be download from http://downloads.dbpedia.org/3.5.1/en/. Specifically, download the "mappingbased_properties_en.nt" file. Rename it to 'facts.gz' and place it as PKG_EXP/DBpedia/facts.gz.

YAGO can downloaded from https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/yago-naga/yago/downloads/. The button is Download YAGO Themes -> CORE -> yagoFacts -> Download TSV. Convert to ".gz" file using gzip if not in this format.




## How to run
Go to code/src/path.py, change the dataset path if needed. The resulted log file name can be set in main.py line 14.

For each dataset, you can run the corresponding command. We recommand to run MetaQA first because DBpedia and YAGO are very large knowledge graph datasets and takes time to finish executing.

```sh
cd code

# quick start with pre-generated query on MetaQA
python main.py --kg MetaQA --method apex apex-n --percent-triples 0.0001 --n_users 1

# generate user queries and test on DBpedia and YAGO
python main.py --kg DBpedia --method apex apex-n --save-queries --percent-triples 0.000001 --n_users 1

python main.py --kg YAGO --method apex apex-n --save-queries --percent-triples 0.000001 --n_users 1
```

If you want to use the same queries stored, change --save-queries to --load-queries. When run with --save-queries, the query files will be re-written, you need to enter "c" when pdb askes whether to proceed.


## Result
For convenience, we save the detailed experiment results in the log files. We record F1 score and running time for each adapting. To quickly view the result, search "Ave Time on Each Training Log" and "Ave Ave F1 on Each Training Log" in the log files. Note that the "Summarizing with ......" under "Ave Ave F1 on Each Training Log" is not a pair. It's the next method applied after the method that generated this result.


## Cite
If you find this repository useful in your research, please consider citing the following paper:

```
@article{li2024apex,
  title={APEX $\^{} 2$: Adaptive and Extreme Summarization for Personalized Knowledge Graphs},
  author={Li, Zihao and Fu, Dongqi and Ai, Mengting and He, Jingrui},
  journal={arXiv preprint arXiv:2412.17336},
  year={2024}
}
```