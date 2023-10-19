from server import GlobalServer
from client import HouseClient
import utils

server = GlobalServer()
client1 = HouseClient()
client2 = HouseClient()

print(f'{server.model.estimators_}\n{client1.local_model.estimators_}\n{client2.local_model.estimators_}')

(error1, accuracy1, best_trees1) = client1.evaluate(server.model)
(error2, accuracy2, best_trees2) = client2.evaluate(server.model)

print(error1, accuracy1)
print(error2, accuracy2)