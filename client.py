import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import uuid

import utils

TREES_BY_CLIENT = 50

class HouseClient():

    def __init__(self) -> None:
        self.id = uuid.uuid4().int
        # Load house data
        (X_train, y_train), (self.X_test, self.y_test) = utils.load_house()

        # Split train set into 10 partitions and randomly use one for training.
        partition_id = np.random.choice(4)
        (self.X_train, self.y_train) = utils.partition(X_train, y_train, 4)[partition_id]

        # Initialize local model and set initial_parameters
        self.local_model = RandomForestRegressor(n_estimators=TREES_BY_CLIENT)
        utils.set_initial_params(self.local_model, X_train, y_train) 
        self.trees = self.local_model.estimators_

    def get_global_parameters(self, global_model: RandomForestRegressor):
            return utils.get_model_parameters(global_model)

    def evaluate(self, global_model: RandomForestRegressor):
        global_model_trees = self.get_global_parameters(global_model)
        
        local_error = mean_absolute_error(self.y_test, self.local_model.predict(self.X_test))
        global_model_error = mean_absolute_error(self.y_test, global_model.predict(self.X_test))

        if local_error < global_model_error:
             error = local_error
            #  print(f'Client_id {self.id}: Erro local de {error} é menor')
        else:
             error = global_model_error
             self.trees = global_model_trees
             utils.set_model_params(self.local_model, self.trees)
            #  print(f'Client_id {self.id}: Erro global de {error} é menor')

        return error, self.trees

    
        
        