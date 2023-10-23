from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

import utils

class FedForest():
    def __init__(self) -> None:
        pass

    def aggregate_fit_best_forest_strategy(self, best_forests: list[RandomForestRegressor]):
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

        utils.set_model_params(self.model, best_forest)
        
        return best_forest