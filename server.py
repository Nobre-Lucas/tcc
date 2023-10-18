from sklearn.ensemble import RandomForestRegressor
import utils

TREES_BY_CLIENT = 1

class GlobalServer():

    def __init__(self, num_rounds=2, num_clients=2) -> None:
        self.model = RandomForestRegressor(n_estimators=TREES_BY_CLIENT)
        X_train, y_train = utils.load_house_server_side()
        utils.set_initial_params(self.model, X_train, y_train)
        self.global_trees = self.model.estimators_
        
        

    

        