from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import NetworkModule, ChartModule
from mesa.visualization.UserParam import UserSettableParameter
import random
import numpy as np


from Model_v2_1_SingleProducer import *

epsilon = np.finfo(np.float64).eps


def network_portrayal(G):
    
    portrayal = dict()
    portrayal["nodes"] = [
        {
            "id": node_id,
            "size": 200 if(agent[0].type=="Producer") else 1,
            "color": "Red" if(agent[0].type=="Producer") else "Green"
        }
        for (node_id, agent) in G.nodes.data("agent")
        
    ]

    portrayal["edges"] = [
        {"id": edge_id, "source": source, "target": target, "color": "Black", "Layer": 0}
        for edge_id, (source, target) in enumerate(G.edges)
    ]
    # for (node_id, agent) in G.nodes.data("agent"):
    #     print(node_id, agent)
    return portrayal


canvas_element = NetworkModule(network_portrayal,800, 800)

chart_element1 = ChartModule([{"Label": "Total Wealth(Producer)",
                      "Color": "Red"},{"Label": "Total Wealth(Consumer)",
                      "Color": "Green"}],
                    data_collector_name='datacollector')
chart_element2 = ChartModule([{"Label": "Gini Coefficient(Consumers)",
                      "Color": "Green"},{"Label": "Gini Coefficient(Producers)",
                      "Color": "Red"}],
                    data_collector_name='datacollector')
chart_element3 = ChartModule([{"Label": "Number of Shut Firms",
                      "Color": "Blue"},{"Label": "Producer Number",
                      "Color": "Orange"}],
                    data_collector_name='datacollector')
chart_element4 = ChartModule([{"Label": "Total Utility",
                      "Color": "Yellow"}],
                    data_collector_name='datacollector')
chart_element5 = ChartModule([{"Label": "Wage Rate",
                      "Color": "Brown"}],
                    data_collector_name='datacollector')
chart_element6 = ChartModule([{"Label": "Leisure Proportion",
                      "Color": "Violet"}],
                    data_collector_name='datacollector')
chart_element7 = ChartModule([{"Label": "Rate of Adjustment",
                      "Color": "Pink"}],
                    data_collector_name='datacollector')
chart_element8 = ChartModule([{"Label": "Excess Labour Demand",
                      "Color": "Magenta"}],
                    data_collector_name='datacollector')
chart_element9 = ChartModule([{"Label": "Value Sum of Absolute Excess Demand",
                      "Color": "Black"}],
                    data_collector_name='datacollector')
# chart_element9 = ChartModule([{"Label": "LabourUtilized",
#                       "Color": "Gold"}],
#                     data_collector_name='datacollector'
#                     )

# chart_element2 = ChartModule([{"Label": "Total Wealth(Producer)",
#                       "Color": "Green"},{"Label": "Total Wealth(Consumer)",
#                       "Color": "Red"}],
#                     data_collector_name='datacollector')

model_params={"Producer_N":UserSettableParameter("slider",
                                                    "Number of Producers",
                                                    10,
                                                    1,
                                                    30,
                                                    5,
                                                    description="Choose how many Producers to include in the model",
                                                ), 
            "Consumer_N": UserSettableParameter("slider",
                                                    "Number of Consumers",
                                                    80,
                                                    50,
                                                    200,
                                                    10,
                                                    description="Choose how many Consumers to include in the model",
                                                ), 
            "InitialConditions": UserSettableParameter("number",
                                                    "Input Seed", 
                                                    value=143
                                                    ),
            "seed": UserSettableParameter("number",
                                        "Model Seed", 
                                        value=1
                                        )                             
            }

server = ModularServer(MonopolyModel,
[canvas_element,chart_element1,chart_element2,chart_element3,chart_element4,chart_element5,chart_element6,chart_element8,chart_element7,chart_element9],
"General Economic Model",model_params=model_params)
server.port = 8521
