import random
import json
import torch
import math
import numpy as np
import os

"""
使用するジェネレーターを今後追加する可能性があるため、class形式で記述をしている
"""

class EpisodicGenerator():
    def __init__(self, datasetName=None, dataset_path=None, max_classes=50, num_elements_per_class=None):
        """
        EpisodicGenerator の初期化
        
        ・Parameters
        datasetName : str
            datasets.json 内のデータセット名
        dataset_path : str
            datasets.json が存在するディレクトリ
        max_classes : int
            エピソードで使用可能な最大クラス数
        num_elements_per_class : list
            各クラスに含まれるサンプル数のリスト
    """
        assert datasetName != None or num_elements_per_class!=None, "datasetName and num_elements_per_class can't both be None"
        
        all_datasets = {}
        self.dataset_path = dataset_path

        if self.dataset_path != '' and self.dataset_path != None:
            json_path = os.path.join(self.dataset_path, 'datasets.json')

            if os.path.exists(json_path):
                f = open(json_path)    
                all_datasets = json.loads(f.read())
                f.close()

        self.datasetName = datasetName
        self.dataset = None
        if datasetName != None and datasetName in all_datasets.keys():
            self.dataset = all_datasets[datasetName]
        if num_elements_per_class == None and self.dataset!=None:
            self.num_elements_per_class = self.dataset["num_elements_per_class"]
        else:
            self.num_elements_per_class = num_elements_per_class
        if self.num_elements_per_class != None:
            self.max_classes = min(len(self.num_elements_per_class), max_classes)

    def select_classes(self, ways):
        """
        エピソードで使用するクラスをランダムに選択する
        """
        n_ways = ways if ways!=0 else random.randint(5, self.max_classes)

        # get n_ways classes randomly
        choices = torch.randperm(len(self.num_elements_per_class))[:n_ways]
        return choices 

    def get_query_size(self, choice_classes, n_queries):
        """
        各クラスの query 数を決める
        """
        if n_queries == 0:
            min_queries = n_queries if n_queries != 0 else 10
            query_size = min([int(0.5*self.num_elements_per_class[c]) for c in choice_classes]) 
            query_size = min(min_queries, query_size)
        else:
            query_size = n_queries
        return query_size

    def get_support_size(self, choice_classes, query_size, n_shots):
        """
        support 全体のサンプル数を決める
        """
        if n_shots == 0:
            beta = 0.
            while beta == 0.:
                beta = torch.rand(1).item()
            support_size = sum([math.ceil(beta*min(100, self.num_elements_per_class[c]-query_size)) for c in choice_classes])
            support_size = min(500, support_size)
        else:
            support_size = len(choice_classes)*n_shots
        return support_size

    def get_number_of_shots(self, choice_classes, support_size, query_size, n_shots):
        """
        各クラスごとの shot 数を決定する
        """
        if n_shots == 0: 
            n_ways = len(choice_classes)
            alphas = torch.Tensor(np.random.rand(n_ways)*(np.log(2)-np.log(0.5))+np.log(0.5)) # sample uniformly between log(0.5) and log(2)
            proportions = torch.exp(alphas)*torch.cat([torch.Tensor([self.num_elements_per_class[c]]) for c in choice_classes])
            proportions /= proportions.sum() # make sum to 1
            n_shots_per_class = ((proportions*(support_size-n_ways)).int()+1).tolist()
            n_shots_per_class = [min(n_shots_per_class[i], self.num_elements_per_class[c]-query_size) for i,c in enumerate(choice_classes)]
        else:
            n_shots_per_class = [n_shots]*len(choice_classes)
        return n_shots_per_class

    def get_number_of_queries(self, choice_classes, query_size, unbalanced_queries):
        """
        各クラスごとの query 数を決定する
        """
        if unbalanced_queries:
            alpha = np.full(len(choice_classes), 2)
            prob_dist = np.random.dirichlet(alpha)
            while prob_dist.min()*query_size*len(choice_classes)<1: # if there is a class with less than one query resample
                prob_dist = np.random.dirichlet(alpha)
            n_queries_per_class = self.convert_prob_to_samples(prob=prob_dist, q_shot=query_size*len(choice_classes))
        else:
            n_queries_per_class = [query_size]*len(choice_classes)
        return n_queries_per_class

    def sample_indices(self, num_elements_per_chosen_classes, n_shots_per_class, n_queries_per_class):
        """
        各クラスから support / query 用インデックスをサンプリングする
        """
        shots_idx = []
        queries_idx = []
        for k, q, elements_per_class in zip(n_shots_per_class, n_queries_per_class, num_elements_per_chosen_classes):
            choices = torch.randperm(elements_per_class)
            shots_idx.append(choices[:k].tolist())
            queries_idx.append(choices[k:k+q].tolist())
        return shots_idx, queries_idx

    def sample_episode(self, ways=0, n_shots=0, n_queries=0, unbalanced_queries=False, verbose=False):
        """
        1エピソードを生成するメイン関数
        """
        choice_classes = self.select_classes(ways=ways)
        
        query_size = self.get_query_size(choice_classes, n_queries)
        support_size = self.get_support_size(choice_classes, query_size, n_shots)

        n_shots_per_class = self.get_number_of_shots(choice_classes, support_size, query_size, n_shots)
        n_queries_per_class = self.get_number_of_queries(choice_classes, query_size, unbalanced_queries)
        shots_idx, queries_idx = self.sample_indices([self.num_elements_per_class[c] for c in choice_classes], n_shots_per_class, n_queries_per_class)

        if verbose:
            print(f'chosen class: {choice_classes}')
            print(f'n_ways={len(choice_classes)}, q={query_size}, S={support_size}, n_shots_per_class={n_shots_per_class}')
            print(f'queries per class:{n_queries_per_class}')
            print(f'shots_idx: {shots_idx}')
            print(f'queries_idx: {queries_idx}')

        return {'choice_classes':choice_classes, 'shots_idx':shots_idx, 'queries_idx':queries_idx}

    def get_features_from_indices(self, features, episode, validation=False):
        """
        episode の index に従って特徴量を抽出する
        """
        choice_classes, shots_idx, queries_idx = episode['choice_classes'], episode['shots_idx'], episode['queries_idx']
        if validation : 
            validation_idx = episode['validations_idx']
            val = []
        shots, queries = [], []
        for i, c in enumerate(choice_classes):
            shots.append(features[c]['features'][shots_idx[i]])
            queries.append(features[c]['features'][queries_idx[i]])
            if validation : 
                val.append(features[c]['features'][validation_idx[i]])
        if validation:
            return shots, queries, val
        else:
            return shots, queries
