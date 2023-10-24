from server import GlobalServer
from client import HouseClient
import utils

# Fase 1: Iniciar o modelo global
server = GlobalServer()

# Fase 2: Inicializar os clientes
client1 = HouseClient()
client2 = HouseClient()
client3 = HouseClient()
client4 = HouseClient()
print(f'{server.model.estimators_}\n{client1.local_model.estimators_}\n{client2.local_model.estimators_}')

for i in range(5):
    # Fase 3: Receber os parâmetros do servidor e avaliar
    print(f'Round: {i+1}')
    (error1, best_trees1) = client1.evaluate(server.model)
    (error2, best_trees2) = client2.evaluate(server.model)
    (error3, best_trees3) = client3.evaluate(server.model)
    (error4, best_trees4) = client4.evaluate(server.model)

    print(f'Client_id {client1.id} - erro absoluto médio: {error1}')
    print(f'Client_id {client2.id} - erro absoluto médio: {error2}')
    print(f'Client_id {client3.id} - erro absoluto médio: {error3}')
    print(f'Client_id {client4.id} - erro absoluto médio: {error4}')

    # Fase 4: Agregar treinamento
    server.aggregate_fit([client1.trees, client2.trees, client3.trees, client4.trees])