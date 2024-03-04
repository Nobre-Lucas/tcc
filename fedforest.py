from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

from scipy.stats import pearsonr

import numpy as np

import random
import warnings
# from sklearn.exceptions import DataConversionWarning

# Suprimir todos os warnings do scikit-learn
# warnings.filterwarnings("ignore", category=DataConversionWarning)

# Se desejar suprimir todos os warnings (não recomendado para depuração)
warnings.filterwarnings("ignore")

import utils

class FedForest():
    def __init__(self, model: RandomForestRegressor) -> None:
        self.model = model

    def aggregate_fit_best_forest_strategy(self, best_forests: list[list[DecisionTreeRegressor]]):
        """
        Essa estratégia percorre as árvores retornadas por cada um dos clientes salvando
        o menor erro entre todas. Ele define a floresta com menor erro como sendo a melhor
        floresta e generaliza ela para toda a rede.
        """
        X_valid, y_valid = utils.load_server_side_validation_data()
        best_forest_error = float('inf') # float('inf') denota um número muito grande
        for forest in best_forests:
            utils.set_model_params(self.model, forest)
            forest_error = mean_absolute_error(y_valid, self.model.predict(X_valid))
            if forest_error < best_forest_error:
                best_forest = forest
                best_forest_error = forest_error

        # print(best_forest)

        utils.set_model_params(self.model, best_forest)
        
        return best_forest
    
    def aggregate_fit_best_trees_strategy(self, best_forests: list[list[DecisionTreeRegressor]]):
        """
        Essa estratégia ordena as árvores de cada floresta com base no seu erro quadrático médio.
        Quanto menor, melhor a árvore.
        Depois, coleta as x% melhores árvores de cada floresta e agrega elas em uma nova floresta
        que se torna o novo modelo global.
        """
        X_valid, y_valid = utils.load_server_side_validation_data()
        best_trees = []
        best_trees_ratio = int(len(best_forests[0]) * 0.5) # numero de melhores arvores por floresta
        print(f'Numero de melhores arvores por floresta é: {best_trees_ratio}')
        for forest in best_forests:
            forest_trees = forest
            trees_sorted = sorted(forest_trees, key=lambda tree: mean_absolute_error(y_valid, tree.predict(X_valid)))
            best_trees.extend(trees_sorted[:best_trees_ratio])
            
        # print(best_trees)

        return best_trees

    def aggregate_fit_best_trees_threshold_strategy(self, best_forests: list[list[DecisionTreeRegressor]], threshold: float):
        """
        Essa estratégia ordena as árvores de cada floresta com base no sua correlação de pearson.
        Em seguida, coleta todas as árvores cujo correlação é maior que um threshold e as agrega em uma nova floresta,
        que se torna o novo modelo global.
        """
        X_valid, y_valid = utils.load_server_side_validation_data()
        best_trees = []

        for forest in best_forests:
            forest_trees = forest
            trees_sorted = sorted(forest_trees, key=lambda tree: pearsonr(y_valid, tree.predict(X_valid)))
            
            # Filtra as árvores com pearson maior que o threshold
            selected_trees = [tree for tree in trees_sorted if pearsonr(y_valid, tree.predict(X_valid))[0] > threshold]
            
            best_trees.extend(selected_trees)

        return best_trees

    
    # def aggregate_fit_random_trees_strategy(self, best_forests: list[list[DecisionTreeRegressor]]):
    #     """
    #     Essa estratégia escolhe um subset aleatório de árvores de cada floresta e agrega essas
    #     árvores em uma nova floresta que se torna o novo modelo global
    #     """
    #     best_trees = []
    #     best_trees_ratio = int(len(best_forests[0]) * 0.5) # numero de melhores arvores por floresta
    #     forest_size = len(forest_trees)

    #     for forest in best_forests:
    #         forest_trees = forest
            
    #         if best_trees_ratio >= forest_size:
    #             # Se o número de árvores a serem selecionadas for maior ou igual ao número de árvores na floresta, 
    #             # apenas adicionamos todas as árvores.
    #             best_trees.extend(forest_trees)
    #         else:
    #             # Caso contrário, escolhemos um conjunto aleatório de árvores.
    #             selected_trees = random.sample(forest_trees, best_trees_ratio)
    #             best_trees.extend(selected_trees)

    #     return best_trees

    def aggregate_fit_random_trees_strategy(self, best_forests):
        best_trees = []
        best_trees_ratio = int(len(best_forests[0]) * 0.5)

        for forest in best_forests:
            forest_trees = np.array(forest)
            num_trees = len(forest_trees)

            if best_trees_ratio >= num_trees:
                best_trees.extend(forest_trees)
            else:
                selected_indices = np.random.choice(num_trees, best_trees_ratio, replace=False)
                selected_trees = forest_trees[selected_indices]
                best_trees.extend(selected_trees)

        return best_trees
        
