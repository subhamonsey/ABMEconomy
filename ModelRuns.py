from Model_v2_1_SingleProducer import MonopolyModel
import matplotlib.pyplot as plt
import numpy as np

model=MonopolyModel(10, 80, 78, 122)
for i in range(1000):
    if model.running:
        model.step()
    else:
        break

print("Process Complete")
modeldata=model.datacollector.get_model_vars_dataframe()
agentdata=model.datacollector.get_agent_vars_dataframe()

eqm_test=[]
def variable_convergence_check(data,window):
    var=data
    var_rolling=var.rolling(window)
    var_rolling_mean=var_rolling.mean()
    d_var=var_rolling_mean.diff()
    d_var=d_var.abs()
    # d_var.plot()
    # plt.show()
    for i in range(500,900):
        if np.allclose(d_var[i:], np.zeros(1000-i +1), atol= 1e-03):
            eqm_test.append(True)
            return
    eqm_test.append(False)
    

# agents=agentdata.set_index(agentdata.index.swaplevel(0,1))
# for i in model.Producers:
#     tempdata=agents.loc[i.id]['Price of commodity']
#     variable_convergence_check(tempdata,100)

variable_convergence_check(modeldata['Excess Labour Demand'],100)
print(eqm_test)
modeldata['Excess Labour Demand'].plot()
plt.show()