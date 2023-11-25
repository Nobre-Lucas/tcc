# Parâmetros das simulações
"""
colunas do CSV:
trees_by_client,round,sim_time,mse1,pearson_corr1,mse2,pearson_corr2,mse3,pearson_corr3,mse4,pearson_corr4
"""
import time

trees_by_client = 10
strategy = 'random'
output_file = f'data/simulation/{strategy}_strategy_sim.csv'

MAX_TREES_NUM = 600
TREES_ITER = 10
# MAX_TREES_NUM = 10
# TREES_ITER = 1

# Fase 1: Iniciar o modelo global
from server import GlobalServer
from client import HouseClient
import utils

server = GlobalServer(trees_by_client)

# Fase 2: Inicializar os clientes
client1 = HouseClient(trees_by_client)
client2 = HouseClient(trees_by_client)
client3 = HouseClient(trees_by_client)
client4 = HouseClient(trees_by_client)
# print(f'{server.model.estimators_}\n{client1.local_model.estimators_}\n{client2.local_model.estimators_}')

while trees_by_client <= MAX_TREES_NUM:
    for round in range(3):
        start_time = time.time()
        # Fase 3: Receber os parâmetros do servidor e avaliar
        print(f'Round: {round+1}')
        (absolute_error1, squared_error1, (pearson_corr1, p_value1), best_trees1) = client1.evaluate(server.model)
        (absolute_error2, squared_error2, (pearson_corr2, p_value2), best_trees2) = client2.evaluate(server.model)
        (absolute_error3, squared_error3, (pearson_corr3, p_value3), best_trees3) = client3.evaluate(server.model)
        (absolute_error4, squared_error4, (pearson_corr4, p_value4), best_trees4) = client4.evaluate(server.model)

        # print(f'Client_id {client1.id} - erro absoluto médio: {absolute_error1} - erro quadrático médio: {squared_error1}')
        # print(f'Client_id {client2.id} - erro absoluto médio: {absolute_error2} - erro quadrático médio: {squared_error2}')
        # print(f'Client_id {client3.id} - erro absoluto médio: {absolute_error3} - erro quadrático médio: {squared_error3}')
        # print(f'Client_id {client4.id} - erro absoluto médio: {absolute_error4} - erro quadrático médio: {squared_error4}')

        print(f'NUM_TREES: {trees_by_client}')
        print(f'Client_id {client1.id} - erro quadrático médio: {squared_error1} - correlação de pearson: {pearson_corr1}')
        print(f'Client_id {client2.id} - erro quadrático médio: {squared_error2} - correlação de pearson: {pearson_corr2}')
        print(f'Client_id {client3.id} - erro quadrático médio: {squared_error3} - correlação de pearson: {pearson_corr3}')
        print(f'Client_id {client4.id} - erro quadrático médio: {squared_error4} - correlação de pearson: {pearson_corr4}')

        # Fase 4: Agregar treinamento
        # server.aggregate_fit([client1.local_model, 
        #                       client2.local_model, 
        #                       client3.local_model, 
        #                       client4.local_model])
        server.aggregate_fit([client1.trees, client2.trees, client3.trees, client4.trees], strategy)

        end_time = time.time()
        with open(output_file, 'a') as out_file:
            execution_time = end_time - start_time
            line = f'{trees_by_client},{round+1},{execution_time},{squared_error1},{pearson_corr1},{squared_error2},{pearson_corr2},{squared_error3},{pearson_corr3},{squared_error4},{pearson_corr4}'
            out_file.write(line)
            out_file.write('\n')

    trees_by_client += TREES_ITER