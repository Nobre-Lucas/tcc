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
# print(f'{server.model.estimators_}\n{client1.local_model.estimators_}\n{client2.local_model.estimators_}')

for i in range(3):
    # Fase 3: Receber os parâmetros do servidor e avaliar
    print(f'Round: {i+1}')
    (absolute_error1, squared_error1, (pearson_corr1, p_value1), best_trees1) = client1.evaluate(server.model)
    (absolute_error2, squared_error2, (pearson_corr2, p_value2), best_trees2) = client2.evaluate(server.model)
    (absolute_error3, squared_error3, (pearson_corr3, p_value3), best_trees3) = client3.evaluate(server.model)
    (absolute_error4, squared_error4, (pearson_corr4, p_value4), best_trees4) = client4.evaluate(server.model)

    # print(f'Client_id {client1.id} - erro absoluto médio: {absolute_error1} - erro quadrático médio: {squared_error1}')
    # print(f'Client_id {client2.id} - erro absoluto médio: {absolute_error2} - erro quadrático médio: {squared_error2}')
    # print(f'Client_id {client3.id} - erro absoluto médio: {absolute_error3} - erro quadrático médio: {squared_error3}')
    # print(f'Client_id {client4.id} - erro absoluto médio: {absolute_error4} - erro quadrático médio: {squared_error4}')

    print(f'Client_id {client1.id} - erro quadrático médio: {squared_error1} - correlação de pearson: {pearson_corr1}')
    print(f'Client_id {client2.id} - erro quadrático médio: {squared_error2} - correlação de pearson: {pearson_corr2}')
    print(f'Client_id {client3.id} - erro quadrático médio: {squared_error3} - correlação de pearson: {pearson_corr3}')
    print(f'Client_id {client4.id} - erro quadrático médio: {squared_error4} - correlação de pearson: {pearson_corr4}')

    # Fase 4: Agregar treinamento
    # server.aggregate_fit([client1.local_model, 
    #                       client2.local_model, 
    #                       client3.local_model, 
    #                       client4.local_model])
    server.aggregate_fit([client1.trees, client2.trees, client3.trees, client4.trees])