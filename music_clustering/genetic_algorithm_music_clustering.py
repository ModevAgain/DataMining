import operator
import numpy as np
import copy
from multiprocessing import *
import queue
import matplotlib.pyplot as plt

class Individual:
        
    'Common base class for all employees'
    def get_data():
        import pandas as pd
        df1 = pd.DataFrame.from_csv("FeatureFileTrainingAllList1.csv")
        df2 = pd.DataFrame.from_csv("FeatureFileTestAllList2.csv")

        import sklearn.preprocessing as skp
        import numpy as np

        for df in [df1,df2]:
            for i in range(0,len(df.columns)):
                df.iloc[:,i] = skp.scale(df.iloc[:,i])
        return [df1,df2]
    
    df1 = get_data()[0]
    df2 = get_data()[1]
    
    def __init__(self, nofattributes,used_attributes,attribute_settings=0,random=True,similarity="euclid"):
        self.nofattributes = nofattributes
        self.used_attributes = used_attributes
        if(random):
            settings = np.zeros(nofattributes)
            for i in range(0,used_attributes):
                success = False
                while(not success):
                    index = np.random.randint(0,self.nofattributes)
                    if(settings[index]==0):
                        settings[index] = 1
                        success=True
            self.attribute_settings = [*settings]
        else:
            self.attribute_settings = attribute_settings
        self.calcFitness(similarity)
   
    def repair(self):
        while(sum(self.attribute_settings)>self.used_attributes):
            #print("REPAIR: deleting random setting")
            self.attribute_settings[np.random.randint(0,self.nofattributes)] = 0
            
        while(sum(self.attribute_settings)<self.used_attributes):
            #print("REPAIR: adding random setting")
            self.attribute_settings[np.random.randint(0,self.nofattributes)] = 1
        
        return self
    
    def cross(self,partner,mutprob,similarity="euclid"):       
        #print("entering cross")
        found_segment = False
        itera = 0
        locus_length = 0
        while(not found_segment):
            itera+=1
            locus_start = np.random.randint(0,self.nofattributes-1)
            locus_end = np.random.randint(locus_start+1,self.nofattributes)
            
            locus_length = locus_end-locus_start
            
            sum_self = sum(self.attribute_settings[locus_start:locus_end])
            sum_partner = sum(partner.attribute_settings[locus_start:locus_end])
            
            if(sum_self == sum_partner and locus_length> int(self.nofattributes/10)):
                found_segment=True
                #print("needed " +str(itera)+ "tries for locus")
        #print("locus length was: "+str(locus_length))        
        settings = copy.deepcopy(self.attribute_settings)
        settings[locus_start:locus_end] = partner.attribute_settings[locus_start:locus_end]
        
        child = Individual(self.nofattributes,self.used_attributes,settings,random=False)
        child.mutate(mutprob)
        #print("calculating child fitness...")
        child.calcFitness(similarity)
        #print("new child fitness is:"+str(child.fitness))
        return child
                  
    def mutate(self,probability):       
        for i in range(0,self.nofattributes):
            if(np.random.rand()<probability):
                if(self.attribute_settings[i]==1):
                    self.attribute_settings[i] = 0
                else:
                    self.attribute_settings[i] = 1
        
        self.repair()
        return self
                
    def calcFitness(self,similarity):
        import operator
        sim_list = {}
        for index1, row1 in self.df1.iterrows():
                inner_dic = {}
                for index2, row2 in self.df2.iterrows():
                    
                    # only used_attributes
                    r1 = row1.tolist()
                    r2 = row2.tolist()
                    
                    rn1 = []
                    rn2 = []
                    for i in range(0,len(self.attribute_settings)):
                        if(self.attribute_settings[i]==1):
                            rn1.append(r1[i])
                            rn2.append(r2[i])
                    
                    
                    if(similarity=="euclid"):
                        value = np.linalg.norm(np.transpose(rn1)-np.transpose(rn2))
                    elif(similarity=="pearson"):
                        from scipy.stats import pearsonr
                        r,p = pearsonr((np.transpose(rn1))     ,     (np.transpose(rn2)))
                        if(r<0):
                            value = 1+r
                        else:
                            value = 1-r
                    else:
                        print("WRONG SIMILARITY OPTION")
                
                    inner_dic[index2] = value
                sorted_list = sorted(inner_dic.items(), key=operator.itemgetter(1))    
                sim_list[index1] = sorted_list
                
                
        ranks = []
        for song in sim_list:
            for rank in range(0,len(sim_list[song])):
                if(sim_list[song][rank][0] == song):
                    ranks.append((song,rank))
        
        
        summ = 0
        for song in ranks:
            summ += song[1]
        self.fitness = summ/len(ranks)
        return self.fitness
		
		
		
class Population:
        
    def __init__(self,nofattributes,used_attributes,popsize,population=[],mutprob=0.01):
        self.popsize = popsize
        self.population = population
        self.nofattributes = nofattributes
        self.used_attributes = used_attributes
        self.mutprob = mutprob
    
    def add(self,ind):
        self.population.append(ind)
    
    def random_populate(self):
        self.population = []
        while(len(self.population)<self.popsize):
            self.add(Individual(self.nofattributes,self.used_attributes))
             
    def getMeanFitness(self):
        summ = 0
        for ind in self.population:
            summ+=ind.fitness
        return summ/len(self.population)
    
    def getMaxFitness(self):
        best = self.population[0].fitness
        best_ind = self.population[0]
        for ind in self.population:
            if(ind.fitness<best):
                best = ind.fitness
                best_ind = ind 
        return(best_ind)
    
    def selectPartners(self):
        partners = []
        
        dic = {}
        for ind in self.population:
            dic[ind] = (1/ind.fitness)
            
        max = sum(dic.values())
        
        for i in range(0,2):
            pick = np.random.uniform(0,max)
            current = 0
            for key, value in dic.items():
                current += value
                if current > pick:
                    partners.append(key)
                    break
                    
        return partners
    
    def getNextGeneration(self):
        new_pop = Population(self.nofattributes,self.used_attributes,self.popsize,population=[])
        
        #print("len(new_pop.population)="+str(len(new_pop.population)))
        #print("new_pop.popsize="+str(new_pop.popsize))
        while(len(new_pop.population) < new_pop.popsize):
            #print("entering while loop - selectin partners...")
            partners = self.selectPartners()
            child = partners[0].cross(partners[1],self.mutprob)
            #print("created child: "+str(child))
            new_pop.add(child)
        
        return new_pop
	
    def createChildren(self,nofchildren,q):
        #children = []
        print("called createChildren")
        for i in range(0,nofchildren):
            partners = self.selectPartners()
            child = partners[0].cross(partners[1],self.mutprob)
            q.put(child)
            #children.append(child)
        #return children

    def getNextGeneration_multicore(self,cores):
        new_pop = Population(self.nofattributes,self.used_attributes,self.popsize,population=[])
        #if __name__ == '__main__':
        if True:
            q = Queue()
            # split popsize into chunks
            chunks = [self.popsize // cores]*cores
            rest = self.popsize - sum(chunks)
            for i in range(0,cores-1):
                if rest>0:
                    chunks[i]+=1
                    rest-=1
            #print("chunks:")
            #print(chunks)
            # create subprocesses
            #print("creating subprocesses")
            processes = []
            for core in range(0,cores):
                processes.append(Process(target=self.createChildren, args=(chunks[core],q)))
			
            #p1 = Process(target=self.createChildren, args=(chunks[0],q))
            #p2 = Process(target=self.createChildren, args=(chunks[1],q))
            #p3 = Process(target=self.createChildren, args=(chunks[2],q))
            #p4 = Process(target=self.createChildren, args=(chunks[3],q))
            #processes = [p1,p2,p3,p4]
            
            for process in processes:
                #print("process started")
                process.start()
            results = []
            while True:
                try:
                    result = q.get(False, 0.01)
                    results.append(result)
                except queue.Empty:
                    pass

                allExited = True
                for t in processes:
                    if t.exitcode is None:
                        allExited = False
                        break
                if allExited & q.empty():
                    break

            for process in processes:
                process.join()

            
            for child in results:
                new_pop.add(child)
            return new_pop

class experiment:
    def __init__(self, nofcores,popsize,used_attributes,iterations,nofattributes=42):
        self.nofcores = nofcores
        self.popsize = popsize
        self.used_attributes = used_attributes
        self.iterations = iterations
        self.mean_fitness = []
        self.alltime_best_fitness = []
        self.current_best_fitness = []
        self.nofattributes = nofattributes

    def start(self):
        p = Population(self.nofattributes,self.used_attributes,self.popsize)
        p.random_populate()
        for iteration in range(0,self.iterations):
            self.mean_fitness.append(p.getMeanFitness())
            current_fitness = p.getMaxFitness()
            self.current_best_fitness.append((current_fitness,current_fitness.fitness))
            if(iteration==0 or current_fitness.fitness < self.alltime_best_fitness[-1][1]):
                self.alltime_best_fitness.append((current_fitness,current_fitness.fitness))
            else:
                self.alltime_best_fitness.append(self.alltime_best_fitness[-1])
            p = p.getNextGeneration_multicore(self.nofcores)
        self.finalpop = p
        
    def plot(self):
        alltime = [x[1] for x in self.alltime_best_fitness]
        curr = [x[1] for x in self.current_best_fitness]

        plt.plot(self.mean_fitness)
        plt.plot(curr)
        plt.plot(alltime)
        plt.ylabel('mean rank')
        plt.xlabel('GA iterations')
        plt.legend(['pop mean fitness', 'current pop best', 'alltime best'], loc='upper left')
        plt.show()