from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

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
    
    def aggregate_fit_random_trees_strategy(self, best_forests: list[list[DecisionTreeRegressor]]):
        best_trees = []
        best_trees_ratio = int(len(best_forests[0]) * 0.5) # numero de melhores arvores por floresta

        for forest in best_forests:
            forest_trees = forest
            
            if best_trees_ratio >= len(forest_trees):
                # Se o número de árvores a serem selecionadas for maior ou igual ao número de árvores na floresta, 
                # apenas adicionamos todas as árvores.
                best_trees.extend(forest_trees)
            else:
                # Caso contrário, escolhemos um conjunto aleatório de árvores.
                selected_trees = random.sample(forest_trees, best_trees_ratio)
                best_trees.extend(selected_trees)

        return best_trees
        
