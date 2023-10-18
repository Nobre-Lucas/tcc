import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

import utils

TREES_BY_CLIENT = 1

class HouseClient():

    def __init__(self) -> None:
        # Load house data
        (X_train, y_train), (self.X_test, self.y_test) = utils.load_house()

        # Split train set into 10 partitions and randomly use one for training.
        partition_id = np.random.choice(2)
        (self.X_train, self.y_train) = utils.partition(X_train, y_train, 2)[partition_id]

        # Initialize local model and set initial_parameters
        self.model = RandomForestRegressor(n_estimators=TREES_BY_CLIENT)
        utils.set_initial_params(self.model, X_train, y_train)
        self.trees = self.model.estimators_

    def get_global_parameters(self, global_model):
            return utils.get_model_parameters(global_model)

    def evaluate(self, global_model):
        global_model_trees = self.get_global_parameters(global_model)
        local_error = mean_absolute_error(self.y_test, self.model.predict(self.X_test))
        utils.set_model_params(self.model, )
        global_model_error = mean_absolute_error(self.y_test, self.model.predict(self.X_test))
        if local_error < global_model_error:
             print('Erro local é menor')
             utils.set_initial_params(self.model, self.trees)
             error = local_error
        else:
             print('Erro global é menor')
             error = global_model_error
             self.trees = global_model_trees
        accuracy = self.model.score(self.X_test, self.y_test)
        return error, len(self.X_test), {"accuracy": accuracy}, self.trees

    
        
        