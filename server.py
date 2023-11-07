from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

import utils
from fedforest import FedForest

TREES_BY_CLIENT = 50

class GlobalServer():

    def __init__(self) -> None:
        self.model = RandomForestRegressor(n_estimators=TREES_BY_CLIENT)
        X_train, y_train = utils.load_house_server_side()
        utils.set_initial_params(self.model, X_train, y_train)
        self.global_trees = self.model.estimators_
        self.strategy = FedForest(self.model)

    def aggregate_fit(self, best_forests: list[RandomForestRegressor]):
        self.model.estimators_ = self.strategy.aggregate_fit_best_trees_strategy(best_forests)
        self.global_trees = self.model.estimators_
        # return self.strategy.aggregate_fit_best_forest_strategy(best_forests)

        
        

    

        