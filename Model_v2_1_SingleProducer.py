from mesa import Model, Agent
from mesa.time import StagedActivation
from mesa.datacollection import DataCollector
from mesa.space import NetworkGrid
#from mesa.batchrunner import batch_run
import networkx as nx
import numpy as np
import random
# import math 
# import matplotlib.pyplot as plt
# import scipy.optimize as sc
# import plotly.graph_objects as go
# import warnings

# warnings.filterwarnings('error')

""" Model feratures:
    1. Consumers are introduced. 
    2. Profit is redistributed among the consumers. 
    3. Utility function is introduced. 
    4. Trade in disequilibrium allowed.
    5. Firms shutting down if negative profit.
    6. Done away with sectors. 
    7. Quantity produced in net.
    8. Market determined single wage rate across firms.
    9. This version considers labour to be homogeneous.
    10. A to be taken as a user settable parameter. 
    11. Naive expectation about profit income. 
    12. Consumers are consuming from the money they earned and recieved in the last time period. 
        This year's labour will determine next year's consumption. 
        So another factor will come for consumers: that of coefficient for future consumption.
    13. Purchases of each period are productions of the previous period.
    14. For the first period, demands are pefectly met. This is initialization period before observation begins.  
    14. Reproducibilty achieved through random generator seed passed in function calls. 
    15. For each of the consumers of a loss-making firm, after shutdown, they either replace the producer and regenerate their 
        objective function or they don't and reassign the coefficients among the remaining providers.(1:1 probability)
    16. Visualization: 
        16.1. A NetworkGrid object of a digraph of agents as nodes, and a directional edge from provider to demander
        16.2. Node-size: proportional to relative wealth rank of agents
        16.3. Charts: Total income, Prices, Excess Demand, Ginni Coefficient, Number of producers

    """
"""
    New features to be added:
    1̶.̶ ̶H̶a̶l̶t̶i̶n̶g̶ ̶c̶o̶n̶d̶i̶t̶i̶o̶n̶ ̶f̶o̶r̶ ̶w̶h̶e̶n̶ ̶e̶q̶u̶i̶l̶i̶b̶r̶i̶u̶m̶ ̶i̶s̶ ̶r̶e̶a̶c̶h̶e̶d̶.̶ 
    2. How to reach equilibrium and characterization of equilibrium condition.  
    3. Figure out the time complexity
    4. Hyperparameters?
    5. Move all_coeffs to Set_ObjectiveFunction
    6̶.̶ ̶C̶o̶n̶s̶i̶d̶e̶r̶ ̶t̶h̶e̶ ̶c̶a̶s̶e̶ ̶o̶f̶ ̶p̶r̶i̶c̶e̶s̶ ̶f̶a̶l̶l̶i̶n̶g̶ ̶t̶o̶ ̶z̶e̶r̶o̶,̶ 
    h̶a̶l̶t̶ ̶b̶y̶ ̶p̶r̶i̶c̶e̶s̶ ̶a̶l̶l̶ ̶f̶a̶l̶l̶e̶n̶ ̶t̶o̶ ̶z̶e̶r̶o̶?̶ ̶A̶l̶s̶o̶ ̶r̶e̶m̶o̶v̶e̶ ̶p̶r̶o̶d̶u̶c̶e̶r̶s̶ ̶w̶i̶t̶h̶ ̶p̶r̶i̶c̶e̶ ̶f̶a̶l̶l̶e̶n̶ ̶t̶o̶ ̶z̶e̶r̶o̶?̶ How to inflate the prices?
    7. Rate of adjustment of wage monotonically decreasing. Address?
    8. Include: a. Halting by max iter. 
                b. Halting by single/zero producer left. 
                c. Halting by high wage and near zero rate of adjustment.
                d. Halting by equilibrium.
    9. Optimize using cython. 
                 
"""

""" Notes to self:(Section to be deleted after done)
    1. D̶o̶u̶b̶l̶e̶ ̶c̶h̶e̶c̶k̶ ̶f̶r̶o̶m̶ ̶c̶o̶n̶s̶u̶m̶e̶r̶ ̶a̶g̶e̶n̶t̶'̶s̶ ̶p̶e̶r̶s̶p̶e̶c̶t̶i̶v̶e̶ ̶a̶l̶l̶ ̶t̶h̶e̶ ̶f̶u̶n̶c̶t̶i̶o̶n̶ ̶d̶e̶f̶i̶n̶i̶t̶i̶o̶n̶s̶.̶
    2. S̶o̶r̶t̶ ̶o̶u̶t̶ ̶t̶h̶e̶ ̶l̶a̶b̶o̶u̶r̶ ̶m̶a̶r̶k̶e̶t̶.̶
    3. R̶e̶s̶e̶t̶t̶i̶n̶g̶ ̶v̶a̶r̶i̶a̶b̶l̶e̶s̶ ̶b̶e̶f̶o̶r̶e̶ ̶n̶e̶x̶t̶ ̶r̶o̶u̶n̶d̶.̶
    4. R̶e̶c̶u̶r̶s̶i̶o̶n̶ ̶e̶r̶r̶o̶r̶ ̶c̶h̶e̶c̶k̶ ̶a̶n̶d̶ ̶e̶x̶c̶e̶p̶t̶i̶o̶n̶ ̶h̶a̶n̶d̶l̶i̶n̶g̶ ̶i̶n̶ ̶g̶e̶n̶e̶r̶a̶l̶.̶
    5. R̶e̶m̶o̶v̶e̶ ̶a̶g̶e̶n̶t̶s̶ ̶f̶r̶o̶m̶ ̶e̶v̶e̶r̶y̶w̶h̶e̶r̶e̶.̶ ̶
    6. N̶e̶t̶w̶o̶r̶k̶G̶r̶i̶d̶ ̶a̶n̶d̶ ̶g̶r̶a̶p̶h̶i̶c̶s̶ ̶v̶i̶s̶u̶a̶l̶i̶z̶a̶t̶i̶o̶n̶ 
    7. D̶a̶t̶a̶c̶o̶l̶l̶e̶c̶t̶o̶r̶.̶
    8. S̶e̶e̶d̶ ̶a̶n̶d̶ ̶r̶e̶p̶r̶o̶d̶u̶c̶a̶b̶i̶l̶i̶t̶y̶.̶
    """


epsilon = np.finfo(np.float64).eps
epsilon

#Hessian Matrix Maxima Minima Check

def MAXIMA_MINIMA_CHECK(HessianMatrix):
    
    LeadingPrincipalMinorsList=[np.linalg.det(HessianMatrix[:(i+1),:(i+1)]) for i in range(len(HessianMatrix))]
    SignCheckList=[i>=0 for i in LeadingPrincipalMinorsList]
    if all(SignCheckList):
        return "Min", LeadingPrincipalMinorsList
    elif all([int(i)==(j)%2  for j,i in enumerate(SignCheckList)]):
        return "Max", LeadingPrincipalMinorsList
    else: 
        return "Degenerate or Saddle", LeadingPrincipalMinorsList
    
    
#Computes the gini coefficient
def compute_gini(wealthlist, model):
    agent_wealths = wealthlist
    x = sorted(agent_wealths)
    N = len(agent_wealths)
    if N<=1:
        return 0
    if sum(x) == 0:
        model.HaltType="ConsumerWealthZero"
        model.datacollector.collect(model)
        model.schedule.steps+=1
        model.running = False
        return 0
    B = sum(xi * (N - i) for i, xi in enumerate(x)) / (N * sum(x))
    return 1 + (1 / N) - 2 * B

#region : Override StagedActivation to allow for model methods

class ModifiedStagedActivation(StagedActivation):
    """A scheduler which allows agent activation to be divided into several
    stages instead of a single `step` method. All agents execute one stage
    before moving on to the next.

    Agents must have all the stage methods implemented. Stage methods take a
    model object as their only argument.

    This schedule tracks steps and time separately. Time advances in fractional
    increments of 1 / (# of stages), meaning that 1 step = 1 unit of time.

    """

    def __init__(
        self,
        model: Model,
        stage_list: list[str] | None = None,
        shuffle: bool = False,
        shuffle_between_stages: bool = False,
    ) -> None:
        """Create an empty Staged Activation schedule.

        Args:
            model: Model object associated with the schedule.
            stage_list: List of strings of names of stages to run, in the
                         order to run them in.
            shuffle: If True, shuffle the order of agents each step.
            shuffle_between_stages: If True, shuffle the agents after each
                                    stage; otherwise, only shuffle at the start
                                    of each step.

        """
        super().__init__(model)
        self.stage_list = ["step"] if not stage_list else stage_list
        self.shuffle = shuffle
        self.shuffle_between_stages = shuffle_between_stages
        self.stage_time = 1 / len(self.stage_list)

    def step(self) -> None:
        """Executes all the stages for all agents."""
        agent_keys = list(self._agents.keys())

        if self.shuffle:
            self.model.random.shuffle(agent_keys)

        for stage in self.stage_list:

            if stage.startswith("model."):
                getattr(self.model,stage[6:])()
                self.time += self.stage_time
                agent_keys = list(self._agents.keys())
                if self.shuffle_between_stages:
                    self.model.random.shuffle(agent_keys)        
                continue

            for agent_key in agent_keys:
                getattr(self._agents[agent_key], stage)()  # Run stage
            # We recompute the keys because some agents might have been removed
            # in the previous loop.
            agent_keys = list(self._agents.keys())
            if self.shuffle_between_stages:
                self.model.random.shuffle(agent_keys)
            self.time += self.stage_time

        self.steps += 1
#endregion

#Agent class where agents' behaviour is defined 
class MonopolyAgent(Agent):

    def __init__(self, id, agent_type, model, K_coeff=None, M_coeff=0 , L_coeff=None, rts=1):
        super().__init__(id, model)
        self.id=id

        self.k=0.3 #Hyperparameter
        self.providers=[] #Gets set in agent __init__
        self.Y=0 #Gets appended during step runs
        self.M=0 #Gets set in model __init__
        self.P=0
        self.L=0
        self.Leisure=0
        self.U=0
        self.revenue=0 #Gets set in model __init__
        #self.RevenueHistory=np.array([])
        self.inventory=0 #Last entry gets set in model __init__. Rest in agent __init__
        self.cost=0
        #self.CostHistory=np.array([])
        self.profit=0
        #self.ProfitHistory=np.array([])
        self.input_prices=np.array([]) #Gets set after first step
        self.QuantityDemanded=0
        self.Excess_Demand=0
        self.Excess_Demand_Value=0
        self.DemandOrder=[]
        self.TheDemand=0
        self.T=365 #Hyperparameter
        self.wL=0
        self.V=0
        self.A=10 #Hyperparameter
        self.shares=np.array([])
        self.RealizedDemand=[]
        self.RemoveAgent=False
        self.GenerateNewObjFunc=False
        #Categorization into "Producer" or "Consumer" according to exogenously given parameters
        self.type=agent_type
        #Returns to scale specification
        self.rts=rts
        #Setting up "broadly" the objective function

        self.K_coeff= self.model.np_rng.uniform(epsilon,1,None) if(K_coeff==None) else K_coeff
        self.L_coeff= self.model.np_rng.uniform(epsilon,1,None) if(L_coeff==None) else L_coeff
        
        if(self.type=='Producer'):
            self.M_coeff= 0
            
            if(self.model.SingleProducerRun):
                self.K_coeff=0
        
        else:
            self.M_coeff= self.model.np_rng.uniform(epsilon,1,None) if(M_coeff==0) else M_coeff

        #Normalization
        temp=np.asarray([self.K_coeff,self.M_coeff,self.L_coeff], dtype='float64')
        temp *= self.rts/np.sum(temp)
        self.K_coeff,self.M_coeff,self.L_coeff=temp
 
    def Set_ObjectiveFunction(self):
    #This function determines the cobb-douglas coefficients of the commodities in the agent's objective function     
        #Set the number of providers in input/consumption basket
        
        self.master_provider_list=self.model.Producers.copy()
        if(self.type=='Producer'):
            self.master_provider_list.remove(self)

        temp=len(self.master_provider_list)
        if((not self.model.SingleProducerRun and temp==0) or (self.type=='Consumer' and temp==0)):
            #print("No possible producer left for ", self.id)
            self.model.HaltType="SingleProducerLeft"
            self.model.datacollector.collect(self.model)
            self.model.schedule.steps+=1
            self.model.running=False
        elif(self.type=='Producer' and self.model.SingleProducerRun):
            self.providers_N=0
            self.provider_coefflist=np.array([])
            self.providers=[]
            self.K_coeff=0
            self.L_coeff=self.rts
        else:
            self.providers_N= temp + 1
            while(self.providers_N > temp):
                self.providers_N= int(self.model.np_rng.uniform(epsilon,temp))+1

            #Sampling out the providers
            
            self.providers=self.random.sample(self.master_provider_list, self.providers_N)
            #Providing the products of each producer a normalized coefficient
            self.provider_coefflist=self.model.np_rng.random((self.providers_N,)) * (1 - epsilon) + epsilon
            self.provider_coefflist*=self.K_coeff/sum(self.provider_coefflist)
        
    def ResetParams(self):
        """Resets the parameters before next step"""

        self.revenue=0
        self.cost=0
        self.profit=0
        self.QuantityDemanded=0
        self.DemandOrder=[]
        self.RealizedDemand=[]
        self.GenerateNewObjFunc=False
        self.Excess_Demand=0
        self.Excess_Demand_Value=0
        self.TheDemand=np.array([])
        self.LabourDemanded=0
        self.LabourOffered=0
        self.L=0
        self.U=0
        self.Y=0
        self.wL=0

    def getPrices(self):
    #This function finds out the prices of the inputs for the calling agent  
        if self.RemoveAgent:
            return
        self.input_prices=np.array([i.P for i in self.providers])

    def Calculate_Demand(self):
    #Gets the optimal bundle maximizing a cobb-douglas function
        '''If the agent is a producer, then maximize the production function, otherwise the utility function for a consumer.
        If rts > 1, the objective function is not concave so the stationary point is not necessarily a maximum. Constrained at 
        budget clearance gives a quasiconcave function being maximized over a convex space and hence the maximum thus attained is
        the global one. For rts < 1, the concave function has a global maxima at its stationary point, for budget less than the cost
        for the maximizing bundle, full budget is exhausted.'''
       
        if self.RemoveAgent:
            return
        
        if(self.type=="Producer"):
            if(self.rts<=1):
                self.all_coeffs=np.copy(self.provider_coefflist)
                self.all_coeffs=np.append(self.all_coeffs,self.L_coeff)
                self.coeff_matrix=np.row_stack([[self.all_coeffs],]*len(self.all_coeffs))
                self.coeff_matrix = self.coeff_matrix - np.eye(len(self.all_coeffs))
                self.all_prices=np.copy(self.input_prices)
                self.all_prices=np.append(self.all_prices,self.model.WageRate)
                b = self.all_prices * np.reciprocal(self.all_coeffs, dtype='float64')
                b/=(self.A*self.P)
                b=np.log(b)
                try:
                    temp=np.linalg.solve(self.coeff_matrix,b)
                    self.ExpectedDemand=np.exp(temp)
                    if(np.allclose(self.ExpectedDemand,np.zeros(len(self.ExpectedDemand)))):
                        if(self.model.InitialRun):
                            self.ResetParams
                            self.Set_ObjectiveFunction()
                            self.getPrices()
                            self.Calculate_Demand()
                            return
                        else:
                            self.RemoveAgent=True
                            #print("Removing agent because lack of optimal bundle", self.id, self.M)
                            return
                    #REVENUE = self.P * self.A * np.prod([pow(k,l) for k,l in zip(self.ExpectedDemand,self.all_coeffs)])
                    COST = np.dot(self.ExpectedDemand, self.all_prices)                    
                    if (COST<=self.M):
                
                        # hessian_dissimilar=np.row_stack([[self.all_coeffs[n]*self.all_coeffs[m]*REVENUE/(self.ExpectedDemand[n] * self.ExpectedDemand[m]) for n in range(len(self.ExpectedDemand))] for m in range(len(self.ExpectedDemand))])
                        # hessian_similar=np.diag([k*REVENUE/l**2 for k,l in zip(self.all_coeffs,self.ExpectedDemand)])
                        # self.HessianMatrix = hessian_dissimilar - hessian_similar
                        # NATURE_OF_CRITICAL_POINT,leadingterms= MAXIMA_MINIMA_CHECK(self.HessianMatrix)[0],MAXIMA_MINIMA_CHECK(self.HessianMatrix)[1] 
                        # if(NATURE_OF_CRITICAL_POINT == 'Max'):
                        self.TheDemand=self.ExpectedDemand[:-1]
                        self.LabourDemanded=self.ExpectedDemand[-1]
                        # else:
                        #     print(f"Not minima, rather {NATURE_OF_CRITICAL_POINT} for id {self.id}, {leadingterms}")
                    
                    else:
                        
                        self.TheDemand=self.provider_coefflist * np.reciprocal(self.input_prices, dtype='float64')
                        self.TheDemand *= self.M/self.rts
                        self.LabourDemanded=self.L_coeff/self.rts * self.M/self.model.WageRate 
                        
                except np.linalg.LinAlgError:

                    raise np.linalg.LinAlgError(f"Solution not present in {self.id}")
                    
            else: 
                    self.TheDemand=self.provider_coefflist * np.reciprocal(self.input_prices, dtype='float64')
                    self.TheDemand *= self.M/self.rts
                    self.LabourDemanded=self.L_coeff/self.rts * self.M/self.model.WageRate

        elif(self.type=="Consumer"):
            self.TheDemand=self.provider_coefflist * np.reciprocal(self.input_prices, dtype='float64')
            self.TheDemand *= self.M/self.K_coeff
            temp=(self.model.WageRate*self.M_coeff*self.T - self.L_coeff*self.V)/(self.model.WageRate*(self.M_coeff + self.L_coeff))
            vprev=self.V
            self.V=0
            self.LabourOffered=temp if(temp>=0) else 0
            if(self.LabourOffered>self.T):
                #print(temp,"   ", self.id)
                #print("Tis me")
                raise ValueError("Labour Offered more than time available", vprev,self.model.WageRate,self.T,vprev + self.model.WageRate*self.T)
            

    def Send_Demand(self):
    #Every producer knows the market demand of their good after this stage

        if self.RemoveAgent:
            return
           
        for j,i in enumerate(self.providers): 
            i.DemandOrder.append((self,self.TheDemand[j]))

    def Sell(self):
        """This function does the computational equivalent of the action of purchase. Adds to the stock of the buyer
        and takes from the stock of the seller. Also it adds to the stock of the money of the seller and takes from the
        stock of the money of the buyer. Note: Income depends on the order of activation of agents in the trade activity."""

        if(self.type=='Producer'):
            self.QuantityDemanded = sum(j for i,j in self.DemandOrder)

            if(self.model.InitialRun):
                return 

            if(self.QuantityDemanded <= self.inventory):
                for i,j in self.DemandOrder:
                    i.RealizedDemand.append((self,j))
                    
                    i.M -= self.P * j
                    i.cost+= self.P * j
                    self.revenue += self.P * j

                self.inventory-=self.QuantityDemanded
                self.M += self.revenue
                #self.RevenueHistory=np.append(self.RevenueHistory,self.revenue)

            else:
                t=self.inventory/self.QuantityDemanded
                for i,j in self.DemandOrder:
                    i.RealizedDemand.append((self,t*j))
                    i.cost += t * self.P * j
                    i.M -= t * self.P * j
                    self.revenue += t * self.P * j

                self.inventory = 0 
                self.M += self.revenue
                #self.RevenueHistory=np.append(self.RevenueHistory,self.revenue)
 
    def AppendCost(self):
        """Function to keep a historic track of purchase amounts"""
        #self.CostHistory=np.append(self.CostHistory,self.cost)
    
    def Produce(self):
        """Production for the next period"""
        if self.RemoveAgent:
            return

        if(self.type=='Producer'):
            t=self.A
            for i,j in self.RealizedDemand:
                t*=pow(j,self.provider_coefflist[self.providers.index(i)])
            t*=pow(self.L,self.L_coeff)
            self.Y=t
            self.inventory+=self.Y

            if(np.allclose(self.inventory,0)):
                self.RemoveAgent= True
                #print(f"Removed by inventory falling to zero|| {self.id} ,{self.M}  ") 
            else:
                False
    
    def Adjust_Price(self):
        """Adjusts price according to classical tatonnement"""
        if self.RemoveAgent:
            return
        
        self.Excess_Demand = self.QuantityDemanded - self.inventory
        self.Excess_Demand_Value = self.Excess_Demand * self.P
        
        if(self.type=='Producer'):

            if np.allclose(self.P, 0):
                #print("Removing producer by price falling to 0.")
                self.RemoveAgent=True
                return
            
            delp=self.k*self.Excess_Demand
            if(self.P + delp <=0 ): 
                self.k*=0.9
                self.Adjust_Price()
            else:
                self.P= self.P + delp 

    def Redistribute_Profit(self):
        """Redistributes the profits of the producers back to consumers and identifies loss-making producers"""
        if(self.type=='Producer' and self.model.schedule.steps >= 1):
            
            self.profit=self.revenue - self.cost
            #self.profit = self.RevenueHistory[-1] - self.CostHistory[-1]
            #self.ProfitHistory.append(self.profit)
            if(self.RemoveAgent==True):
                return
            if(self.profit>0):
                for j,i in enumerate(self.shares):
                    self.model.schedule._agents[(1,j)].V += self.profit *(1-self.model.PRR)* i
                self.M-=(1-self.model.PRR)*self.profit

    def CalculateUtility(self):
        """Utility this period through consumption, money earned and leisure"""
        if(self.type=='Consumer'): 
            t=self.A
            for i,j in self.RealizedDemand:
                t*=pow(j,self.provider_coefflist[self.providers.index(i)])
            t*=pow(self.wL + self.V, self.M_coeff)
            self.Leisure = self.T - self.L
            t*=pow(self.Leisure,self.L_coeff)
            self.U=t

    def Calculate_Income(self):
        """"Consumer income= wL +V"""

        if(self.type=='Consumer'):
            self.M += self.V

        if(np.allclose(self.M,0) or self.M < 0):
            #print("Negative income for agent ", self.id, " Income= ", self.M)
            self.M=0

    def Adjust_ObjectiveFunction(self):
        """Adjusts the objective function for demanders of producers shutting down. Either replaces(p=0.5) or removes(p=0.5) the outgoing
        producer in their objective function"""

        if(not self.RemoveAgent):
            providers_to_be_removed=[]
            for j,i in enumerate(self.providers):
                if(i.RemoveAgent):
                    self.master_provider_list.remove(i)
                    replace=bool(self.random.getrandbits(1)) if(False in [a.RemoveAgent for a in self.master_provider_list if a not in self.providers]) else False
                    if(replace):
                        temp=self.provider_coefflist[j]
                        self.provider_coefflist[j]=self.model.np_rng.uniform(epsilon,1,None)
                        self.provider_coefflist*=self.K_coeff/np.sum(self.provider_coefflist)
                        self.providers[j]=self.random.choice([a for a in self.master_provider_list if (a not in self.providers) and (not a.RemoveAgent)])
                        try:
                            self.model.grid.G.add_edge(self.providers[j].pos,self.pos)
                        except:
                            #print(self.providers[j].id,self.pos)
                            raise ValueError("Can't add edge")
                        self.model.G.add_edge(self.providers[j].pos,self.pos)

                    else:
                        providers_to_be_removed.append(i)
        
            for i in providers_to_be_removed:
            
                temp=self.providers.index(i)
                self.providers.remove(i) 
                self.provider_coefflist=np.delete(self.provider_coefflist, temp)
                self.providers_N -= 1
                if(self.providers_N<=0 and not(self.model.SingleProducerRun and self.type== 'Producer')): 
                    self.GenerateNewObjFunc=True
                else:
                    self.provider_coefflist*=self.K_coeff/np.sum(self.provider_coefflist)
            
            if len(self.model.Producers) <=1:
                self.model.HaltType="SingleProducerLeft"
                self.model.datacollector.collect(self.model)
                self.model.steps+=1
                self.model.running=False
    def Remove_Loss_Makers(self):
        
        """Removes the zero inventory firms from all relevant lists"""
        if(self.RemoveAgent):
            for j,i in enumerate(self.shares):
                if(self.M<0):
                    raise ValueError("Income less than zero", self.M, self.id)
                self.model.schedule._agents[(1,j)].V += self.M * i
            self.M=0
            #print(f"Removing {self.id} || time {self.model.schedule.time} || step {self.model.schedule.steps}")
            self.model.LossMakingProducer.append(self)
            self.model.Producers.remove(self)
            self.model.schedule.remove(self)
            self.model.grid.remove_agent(self)
            k,l= self.id
            self.model.G.remove_node(k*self.model.Producer_N_initial + l)
            
            for i in self.model.schedule.agents:
                if self in i.master_provider_list:
                    i.master_provider_list.remove(self)

            if(self.model.Producer_N >= 2):
                self.model.Producer_N -= 1
                
            if len(self.model.Producers) <= 1:
                self.model.HaltType="SingleProducerLeft"
                self.model.datacollector.collect(self.model)
                self.model.schedule.steps+=1
                self.model.running=False
    
    def RegenerateObjectiveFunction(self):
        """Objecive function regenerated for demanders with zero providers"""

        if(self.GenerateNewObjFunc or self.providers_N<=0):
            self.Set_ObjectiveFunction()
            for prov in self.providers:
                self.model.grid.G.add_edge(prov.pos, self.pos)
                self.model.G.add_edge(prov.pos, self.pos)

class MonopolyModel(Model):

    def __init__(self, Producer_N, Consumer_N, InitialConditions = None,seed=None):
        
        self.Producer_N_initial=Producer_N
        #To initialize the model object with exogenously given parameters
        if(InitialConditions == None or type(InitialConditions)== int):
            #Randomly generate initial conditions
            pn=Producer_N
            cn=Consumer_N
            sd_input= InitialConditions
            print("InitCond Seed=",InitialConditions)
            random.seed(sd_input)
            rng=np.random.default_rng(sd_input)
            
            PM=np.full(pn,1000000,dtype='float64')
            CM=np.full(cn,1000,dtype='float64')
            
            PP=rng.uniform(epsilon,10000,pn)
            W=1000*3/100 

            rtsp=np.absolute(rng.normal(0.9,0.6,pn))
            rtsc=np.full(cn,1)

            sharesmatrix=[]
            for i in range(0,pn):
                shareholders=random.sample(list(range(0,cn)),int(rng.uniform(1,cn)))
                shareholdershare=[rng.uniform(epsilon,1) if i in shareholders else 0 for i in range(0,cn)]
                shareholdershare=[i/sum(shareholdershare) for i in shareholdershare]
                sharesmatrix.append(shareholdershare)

            sharesmatrix=np.array(sharesmatrix)

            inputdictionary={'M_P': PM, 'M_C': CM, 'P_P': PP,'P_C': W, 'RTS_P': rtsp, 'RTS_C': rtsc, 'ES':sharesmatrix}
            self.InitialConditions=inputdictionary
        elif(isinstance(InitialConditions,dict)):
        #Explicitly set initialconditions
            self.InitialConditions=InitialConditions
        
        self.SingleProducerRun= True if Producer_N==1  else False    
        
        print("Model Seed=", seed)
        self.random.seed(seed)
        self.np_rng=np.random.default_rng(seed)  
        self.Producer_N=Producer_N
        self.Consumer_N=Consumer_N

        self.num_agents=self.Producer_N + self.Consumer_N
        self.Producers=[]
        self.Consumers=[]
        self.NoDemandProducer=[]
        self.LossMakingProducer=[]
        self.WageRate=0
        self.TotalWealth=0
        self.LabourDemand=0
        self.LabourSupply=0

        self.k=5e-04 #Hyperparameter
        self.PRR= 0.9 #Hyperparameter
        self.HaltType="Disequilibrium"

        #StagedSchedule
        stage_list=["ResetParams" , "getPrices", "Calculate_Demand", "Send_Demand", "Sell", "model.LabourMarketTransact", 
        "Produce", "Adjust_Price", "model.AdjustWageRate","Redistribute_Profit", "CalculateUtility", "Adjust_ObjectiveFunction", 
        "Remove_Loss_Makers", "Calculate_Income", "RegenerateObjectiveFunction", "model.CalculateRelevantVariables"]
        self.schedule = ModifiedStagedActivation(self,stage_list,shuffle=True)

        #region : Create the agents
        #Producers
        for i in range(0,self.Producer_N):
            a=MonopolyAgent((0,i),'Producer',self,rts=self.InitialConditions['RTS_P'][i])
            self.Producers.append(a)
            self.schedule.add(a)
        
        #Consumers
        for i in range(0,self.Consumer_N):
            try:
                a=MonopolyAgent((1,i),'Consumer',self,rts=self.InitialConditions['RTS_C'][i])
            except:
                raise IndexError(f"{i}") 
            self.Consumers.append(a)
            self.schedule.add(a)
        
        #endregion

        #Generate the agents' objective function
        for i in self.schedule.agents:
            i.Set_ObjectiveFunction()

        #region : Network creation 
        #NetworkX object creation  
        self.G=nx.DiGraph()
        self.G.add_nodes_from(self.schedule.agents)
        for i in self.schedule.agents:
            for j in i.providers:
                self.G.add_edge(j,i)
        
        #Mesa NetworkGrid object creation
        self.grid=NetworkGrid(self.G)
        for i in self.grid.G.nodes():
            #self.grid.G.nodes[i]["agent"]=[]
            self.grid.G.nodes[i]["agent"].append(i)
            k,l=i.id
            i.pos= k * self.Producer_N_initial + l 
        self.grid.G= nx.convert_node_labels_to_integers(self.grid.G,label_attribute="agent")
        for i in self.grid.G.nodes():
            self.grid.G.nodes[i]["agent"]=[self.grid.G.nodes[i]["agent"]]     

        
        nx.set_node_attributes(self.grid.G,nx.kamada_kawai_layout(self.grid.G),'pos')
        self.G=self.grid.G 
        #print("Network created")
        #endregion

        self.RemoveNoDemandProducers()
        
        self.running=True
            
        self.Set_InitialConditions(d=self.InitialConditions)
        self.Inventory_Initialization()
        print("Number of producers=", len(self.Producers))
        print("Number of consumers=", len(self.Consumers))

        self.datacollector=DataCollector(model_reporters={"Total Wealth":"TotalWealth", 
        "Total Wealth(Producer)": "TotalWealth_P", "Total Wealth(Consumer)": "TotalWealth_C",
        "Producer Number": "Producer_N", "Consumer Number": "Consumer_N", "Number of Shut Firms": "LossMakingProducer_N",
        "Gini Coefficient(Consumers)": "GiniCoefConsumers","Gini Coefficient(Producers)": "GiniCoefProducers", "Total Utility": "TotalUtility",
        "Value Sum of Absolute Excess Demand":"AggregateExcessDemand", "Wage Rate":"WageRate", "Rate of Adjustment": "k", 
        "Leisure Proportion": "LeisureProportion", "Excess Labour Demand":"ExcessLabourDemand", "Halt Type":"HaltType"} , 
        agent_reporters={"Rate of Adjustment": "k", "Produce": "Y", "Wealth": "M", "Price of commodity": "P", "Labour Sold/Bought": "L",
        "Utility": "U", "Revenue": "revenue", "Cost": "cost", "Inventory": "inventory", "Profit": "profit", 
        "Quantity Demanded" : "QuantityDemanded", "Profit Income": "V", "Wage Income": "wL" , "RTS": "rts", "K_coeff": "K_coeff", "L_coeff": "L_coeff",
        "Number of providers": "providers_N", "Leisure": "Leisure"})
        self.CalculateRelevantVariables()
        self.datacollector.collect(self)
    
    def CalculateRelevantVariables(self):

        self.TotalWealth=sum(i.M for i in self.schedule.agents)
        self.TotalWealth_P=sum(i.M for i in self.Producers)
        self.TotalWealth_C=sum(i.M for i in self.Consumers)
        self.LossMakingProducer_N=len(self.LossMakingProducer)
        self.AggregateExcessDemand=sum(abs(i.Excess_Demand_Value) for i in self.Producers)
        self.TotalUtility=sum(i.U for i in self.Consumers)
        self.GiniCoefConsumers=compute_gini([i.M for i in self.Consumers], self)
        self.GiniCoefProducers=compute_gini([i.M for i in self.Producers], self)
        self.LeisureProportion=sum(i.Leisure for i in self.Consumers)/sum(i.T for i in self.Consumers) *100
        self.ExcessLabourDemand=self.LabourDemand - self.LabourSupply
        
        # self.LabourUtilized=  0 if(self.schedule.steps<2) else min(self.LabourDemand, self.LabourSupply)/self.LabourSupply * 100

    def AdjustWageRate(self):
        """The Economy adjusting wagerate to equilibriate supply and demand"""
        self.delp= self.k * (self.LabourDemand - self.LabourSupply)
        if(self.WageRate + self.delp <=0):
            self.k*=0.9
            self.AdjustWageRate()
        else:
            self.WageRate= self.WageRate + self.delp

    
    def LabourMarketTransact(self):
        """The Economy allocating work to labourers in a centralized manner"""    
        self.LabourSupply=sum(i.LabourOffered for i in self.Consumers) 
        self.LabourDemand=sum(i.LabourDemanded for i in self.Producers)

        if(self.LabourSupply<= self.LabourDemand):
            temp=self.LabourSupply/self.LabourDemand
            for i in self.Producers:
                i.L=temp*i.LabourDemanded
                i.M -= i.L*self.WageRate
                i.cost+=i.L*self.WageRate
                 
            for i in self.Consumers:
                i.L=i.LabourOffered   
                i.M += i.L*self.WageRate
                i.wL= i.L*self.WageRate

        else:
            temp=self.LabourDemand/self.LabourSupply
            for i in self.Producers:
                i.L=i.LabourDemanded
                i.M -= i.L*self.WageRate 
                i.cost+=i.L*self.WageRate

            for i in self.Consumers:
                i.L=temp * i.LabourOffered
                i.M += i.L*self.WageRate
                i.wL= i.L*self.WageRate

    def Inventory_Initialization(self):
        """The inventory stock from which purchases are done in step 0"""

        self.InitialRun=True
        for i in self.schedule.agents:
            i.getPrices()
        
        for i in self.schedule.agents:
            i.Calculate_Demand()
        
        for i in self.schedule.agents:
            i.Send_Demand()
        
        for i in self.Producers:
            i.QuantityDemanded = sum(j for f,j in i.DemandOrder)
            i.inventory=i.QuantityDemanded
            
        self.InitialRun=False

    def RemoveNoDemandProducers(self):
        """Removes any producer agent created with no takers"""
        for i in self.Producers:
            present=False
            for j in self.schedule.agents:
                if i in j.providers:
                    present=True
                    break

            if(present==False):
                self.RemoveAgent=False
                self.NoDemandProducer.append(self)
                #print(f"Removed agent {i.id} by nonzero demand intital condition.")
            
    def Set_InitialConditions(self, d:dict): 
        """This function initializes all the agents with the speciified initial conditions. Initial conditions to be provided
            in a dictionary form. 
        
            Initial Conditions to be specified as lists: 
            1. Initial money endowment of producers(M_P)
            2. Initial money endowment of consumers(M_C) 
            3. Initial commodity prices(P_P) 
            4. Initial wage rate(P_C)
            5. Return to scales of producers(RTS_P)
            6. return to scales of consumers(RTS_C)
            7. Equity(profit) shares of consumers(ES) 
            
            Format: All parameters except 4 and 7 to be given in unidimensional indexable-array-like format
            4: Double 
            7: 2-dimensional indexable-array-like format preferably np.ndarray of shape (Producer_N, Consumer_N)"""

        #1    
        for j,i in enumerate(self.Producers):
            i.M=d['M_P'][j]
        #2
        for j,i in enumerate(self.Consumers):
            i.M=d['M_C'][j]
        #3
        for j,i in enumerate(self.Producers):
            i.P=(d['P_P'][j])
        #4
        self.WageRate=d['P_C']

        #7 
        for j,i in enumerate(self.Producers):
            i.shares=d['ES'][j]
           
    def variable_convergence_check(self,data,window):
        var=data
        var_rolling=var.rolling(window)
        var_rolling_mean=var_rolling.mean()
        d_var=var_rolling_mean.diff()
        d_var=d_var.abs()
        # d_var.plot()
        # plt.show()
        for i in range(500,900):
            if np.allclose(d_var[i:], np.zeros(d_var[i:].shape), atol= 1e-03):
                return True
        return False
        
    def step(self):
        self.schedule.step()
        if self.schedule.steps==1000:
            
            eqm_test=[]
            modeldata=self.datacollector.get_model_vars_dataframe()
            eqm_test.append(self.variable_convergence_check(data=modeldata['Wage Rate'], window=100))
            agentdata=self.datacollector.get_agent_vars_dataframe()
            agents=agentdata.set_index(agentdata.index.swaplevel(0,1))
            for i in self.Producers:
                tempdata=agents.loc[i.id]['Price of commodity']
                eqm_test.append(self.variable_convergence_check(tempdata,100))
            
            if all(eqm_test):
                self.HaltType="Equilibrium"
                self.datacollector.collect(self)
        self.datacollector.collect(self)



#Batch Run

# params={"Producer_N":10,"Consumer_N":80,"InitialConditions":range(100),"seed":range(100,200)}

# results = batch_run(
#     MonopolyModel,
#     parameters=params,
#     iterations=1,
#     max_steps=150,
#     number_processes=1,
#     data_collection_period=-1,
#     display_progress=True,
# )

# region : Scratch pad

# model1=MonopolyModel(Producer_N=10, Consumer_N=80, InitialConditions= 8, seed= 152)

# # # # # # # sum([i.rts>1 for i in model1.Producers])
# # # #nx.draw(model1.grid.G)
# # # #plt.show()

# for i in range(150):
#     print(i)
#     model1.step()

# model1.Producers[0].rts

# [(a.rts,a.id) for a in model1.Producers]
# nx.set_node_attributes(model1.grid.G,nx.kamada_kawai_layout(model1.grid.G),'pos')

# edge_x = []
# edge_y = []
# for edge in model1.grid.G.edges():
#     x0, y0 = model1.grid.G.nodes[edge[0]]['pos']
#     x1, y1 = model1.grid.G.nodes[edge[1]]['pos']
#     edge_x.append(x0)
#     edge_x.append(x1)
#     edge_x.append(None)
#     edge_y.append(y0)
#     edge_y.append(y1)
#     edge_y.append(None)

# edge_trace = go.Scatter(
#     x=edge_x, y=edge_y,
#     line=dict(width=0.5, color='#888'),
#     hoverinfo='none',
#     mode='lines')

# node_x = []
# node_y = []
# for node in model1.grid.G.nodes():
#     x, y = model1.grid.G.nodes[node]['pos']
#     node_x.append(x)
#     node_y.append(y)

# node_size=[]
# for agent in model1.grid.G.nodes.data('agent'):
#     node_size.append(30 if agent[1][0].type == 'Producer' else 15)


# node_trace = go.Scatter(
#     x=node_x, y=node_y,
#     mode='markers',
#     hoverinfo='text',
#     marker=dict(
#         showscale=True,
#         # colorscale options
#         #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
#         #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
#         #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
#         colorscale='YlGnBu',
#         reversescale=True,
#         color=[],
#         size=node_size,
#         colorbar=dict(
#             thickness=15,
#             title='Agent Wealth',
#             xanchor='left',
#             titleside='right'
#         ),
#         line_width=2))

# node_wealth = []
# node_text = []
# for node, agent in enumerate(model1.grid.G.nodes.data('agent')):
#     node_wealth.append(agent[1][0].M)
#     node_text.append('Agent Id:'+str(agent[1][0].id) + 'Wealth: '+str(agent[1][0].M))

# node_trace.marker.color = node_wealth
# node_trace.text = node_text

# fig = go.Figure(data=[edge_trace, node_trace],
#              layout=go.Layout(
#                 title='<br>Network Economy',
#                 titlefont_size=16,
#                 showlegend=False,
#                 hovermode='closest',
#                 margin=dict(b=20,l=5,r=5,t=40),
#                 annotations=[ dict(
#                     showarrow=False,
#                     text=f"Producers:{model1.Producer_N} \n Consumers:{model1.Consumer_N}", 
#                     xref="paper", yref="paper",
#                     x=0.005, y=-0.002 ) ],
#                 xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#                 yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
#                 )
# fig.show()

# G = nx.random_geometric_graph(200, 0.125)
# random.seed(143)
# rng=np.random.default_rng(143)
# PM=[1000000] *pn
# CM=[1000] * cn

# PP=[rng.uniform(epsilon,100) for i in range(0,pn)]
# W=np.mean(CM)*6/10 

# rtsp=[abs(rng.normal(0.9,0.6)) for i in range(0,pn)]
# rtsc=[rng.uniform(0.7,0.95) for i in range(0,cn)]

# sharesmatrix=[]
# for i in range(0,pn):
#     print(i)
#     shareholders=random.sample(list(range(0,cn)),int(rng.uniform(epsilon,cn-1))+1)
#     shareholdershare=[rng.uniform(epsilon,1) if i in shareholders else 0 for i in range(0,cn)]
#     shareholdershare=[i/sum(shareholdershare) for i in shareholdershare]
#     sharesmatrix.append(shareholdershare)

# sharesmatrix=np.array(sharesmatrix)

# inputdictionary={'M_P': PM, 'M_C': CM, 'P_P': PP,'P_C': W, 'RTS_P': rtsp, 'RTS_C': rtsc, 'ES':sharesmatrix}

# model1=MonopolyModel(10,80,14,1)

# [i.TheDemand for i in model1.schedule.agents]

# ExpectedProfitsConstrainedEquality=[]
# for i in model1.Producers:
#     REVENUE = i.P * i.A * np.prod([pow(k,l) for k,l in zip(i.TheDemand,i.provider_coefflist)]) * pow(i.LabourDemanded, i.L_coeff)
#     COST=sum(g*h for g,h in zip(i.TheDemand,i.input_prices)) + i.LabourDemanded*i.model.WageRate
#     PROFIT= REVENUE - COST
#     ExpectedProfitsConstrainedEquality.append((i,PROFIT))

# ExpectedProfitsUnconstrained=[]
# for i in model1.Producers:
#     coeff=i.provider_coefflist.copy()
#     coeff.append(i.L_coeff)
#     A=np.row_stack([coeff for j in coeff])
#     A= A - np.eye(len(coeff))
#     prices=i.input_prices.copy()
#     prices.append(i.model.WageRate)
#     x=np.array([np.log(k/(i.A*i.P*l)) for k,l in zip(prices,coeff)])
#     try:
#         temp2=np.linalg.solve(A,x)
#         demand=np.exp(temp2)
#         REVENUE = i.P * i.A * np.prod([pow(k,l) for k,l in zip(demand,coeff)])
#         COST =sum(g*h for g,h in zip(demand,prices))
#         PROFIT= REVENUE - COST
                
#         hessian_dissimilar=np.row_stack([[coeff[n]*coeff[m]*REVENUE/(demand[n] *demand[m]) for n in range(len(demand))] for m in range(len(demand))])
#         hessian_similar=np.diag([k*REVENUE/l**2 for k,l in zip(coeff,demand)])
#         i.HessianMatrix=hessian_dissimilar - hessian_similar
    
#         NATURE_OF_CRITICAL_POINT= MAXIMA_MINIMA_CHECK(i.HessianMatrix)[0]
        
#         ExpectedProfitsUnconstrained.append((i,PROFIT,COST<=i.M,NATURE_OF_CRITICAL_POINT))
        
            
#     except np.linalg.LinAlgError:

#         print(f"Solution not present in {i.id}")


# [(i.id,i.rts,j) for i,j in ExpectedProfitsConstrainedEquality]
# for i,p,c,n in ExpectedProfitsUnconstrained:
#     print(f"Id={i.id}, RTS={i.rts}, PROFIT= {p}, COST<=M :{c}, Nature of critical point= {n}") 



# for (node_id, agent) in model1.G.nodes.data("agent"):
#         print(node_id, agent)


# a=[model1.grid.G.nodes[i]["agent"] for i in model1.grid.G.nodes()]

# nx.draw(model1.G)
# plt.show()

# zerod=[model1.G.degree[i] for i in model1.G.nodes()]

# a=[model1.network.G.nodes]


# for i in range(300):
#     print(model1.schedule.steps)
#     model1.step()

# data=model1.datacollector.get_model_vars_dataframe()
# agentdata=model1.datacollector.get_agent_vars_dataframe()
# agentdata.to_pickle("AGENT Producer_N=10, Consumer_N=80, InitialConditions= 243134, seed= 1241241.pkl")
# data.to_pickle("MODEL Producer_N=10, Consumer_N=80, InitialConditions= 243134, seed= 1241241.pkl")

# data.to_pickle("Run Outputs\Model 10.80.143.1 v6.pkl")
# agentdata.to_pickle("Run Outputs\Agent 10.80.143.1 v6.pkl")

# model1.G._node[14]['agent'][0].providers_N

#endregion

# print(MAXIMA_MINIMA_CHECK(np.array([[3,0,3],
#                         [0,1,-2],
#                         [3,-2,8]])))

# print(np.linalg.det(np.array([[1,4,6],
#                         [4,2,1],
#                         [6,1,6]])))
# f=lambda x: -1*(572.45893818264705 * pow(x[0],0.495113057618944) * pow(x[1],0.01680168482152243) - 83.09833188891115*x[0] - 30*x[1])
# a=sc.basinhopping(f,[11.4,1.075])
# a.x
# a.fun
# a.message


