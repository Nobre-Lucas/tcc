from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import utils

TREES_BY_CLIENT = 470

class GlobalServer():

    def __init__(self, num_rounds=2, num_clients=2) -> None:
        self.model = RandomForestRegressor(random_state=42, 
                                           n_estimators=TREES_BY_CLIENT, 
                                           max_depth=27, 
                                           max_leaf_nodes=6121)
        X_train, y_train = utils.load_house_server_side()
        utils.set_initial_params(self.model, X_train, y_train)
        self.global_trees = self.model.estimators_

    def aggregate_fit(self, best_forests: list[RandomForestRegressor]):
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

        
        

    

        