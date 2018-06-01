import operator
import numpy as np
import copy
from multiprocessing import *
import queue
import matplotlib.pyplot as plt
import time
from scipy.stats import pearsonr
import operator

class Individual:
# a individual "creature" in the genetic algorithm concept
# It has following attributes:
# nofattributes = how many features/settings are there? In the music clustering we can use 42 features for clustering

# used_attributes = how many features/settings this individual uses out of the total nofattributes (e.g. 15 out of 42)

# attribute_settings = a numerical list with length=nofattributes and sum(list)=used_attributes. Each element in the list corresponds to an # extracted feature (= column in dataframe). attribute_settings[0] refers to the first column in the dataframe and so on. Valid elements in # the list are 0 and 1. 0 means this feature is turned off/ignored, 1 means this feature is used for fitness calculation and clustering.

# similarity = "euclid" or "pearson". This sets the similarity measure, which is used to calculate the fitness of the individual.

# random = if True the individual is initialized with random attribute_settings. If False an attribute_settings list with 0 and 1 can be  #passed to the class constructor to initialize an individual with desired attribute_settings.

    def get_data():
    # this function loads the FeatureFileTraining-csv files once as static class variables in form of a pandas dataframe
        import pandas as pd
        df1 = pd.DataFrame.from_csv("FeatureFileTrainingAllList1.csv")
        df2 = pd.DataFrame.from_csv("FeatureFileTestAllList2.csv")

        import sklearn.preprocessing as skp
        import numpy as np

        for df in [df1,df2]:
            for i in range(0,len(df.columns)):
                df.iloc[:,i] = skp.scale(df.iloc[:,i])
        return [df1,df2]
    
    # set static class variables
    df1 = get_data()[0]
    df2 = get_data()[1]
    
    def __init__(self, nofattributes,used_attributes,similarity,attribute_settings=0,random=True):
    # class constructor
        self.nofattributes = nofattributes
        self.used_attributes = used_attributes
        self.similarity = similarity
        if(random):
            # perform random initialization of attribute_settings
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
            # use attribute_settings passed to the constructor
            self.attribute_settings = attribute_settings
        # calculate fitness of this individual
        self.calcFitness()
   
    
    def cross(self,partner,mutprob):       
        found_segment = False
        locus_length = 0
        # start search for matching segments in both individuals
        while(not found_segment):
            # randomly pick 2 segment indices (segment start and segment end)
            locus_start = np.random.randint(0,self.nofattributes-1)
            locus_end = np.random.randint(locus_start+1,self.nofattributes)
            locus_length = locus_end-locus_start
            
            # compare number of attributes set in the segment in self and mating partner
            sum_self = sum(self.attribute_settings[locus_start:locus_end])
            sum_partner = sum(partner.attribute_settings[locus_start:locus_end])
            
            # if the sums are the same and the locus_lenght is at least a 10th of the DNA-string accept segment
            if(sum_self == sum_partner and locus_length> int(self.nofattributes/10)):
                found_segment=True

        # copy the settings from self individual       
        settings = copy.deepcopy(self.attribute_settings)
        # replace the segment from self with the partners segment
        settings[locus_start:locus_end] = partner.attribute_settings[locus_start:locus_end]
        
        # create the child with the new crossed attribute settings
        child = Individual(self.nofattributes,self.used_attributes,self.similarity,attribute_settings=settings,random=False)
        # let the child mutate with a probability of mutprob
        child.mutate(mutprob)
        # calculate the fitness of the child
        child.calcFitness()
        return child
                  
    def mutate(self,probability):
        # this function iterates over every gene (bit) in the DNA (attribute_settings) and flips it with a certain probabilty
        # If a flip is performed, the number of set attributes doesn't match the self.used_attributes anymore. 
        # Therefore an inverse flip must be applied somewhere else in the DNA.
        for i in range(0,self.nofattributes):
            # roll for mut probability 
            if(np.random.rand()<probability):
                # if probability is hit start mutation for gene
                action = "none"
                # if attribute is 1 set to 0 and remember decrease (action="dec")
                # if attribute is 0 set to 1 and remebmer increase (action="inc")
                if(self.attribute_settings[i]==1):
                    self.attribute_settings[i] = 0
                    action ="dec"
                else:
                    self.attribute_settings[i] = 1
                    action ="inc"
                    
                #search for random bit which can be inversed to counter increase or decrease action
                repaired = False
                while(not repaired):
                    idx = np.random.randint(self.nofattributes-1)
                    
                    if(idx!=i):
                        if(self.attribute_settings[idx]==0 and action=="dec"):
                            self.attribute_settings[idx]=1
                            repaired = True
                        elif(self.attribute_settings[idx]==1 and action=="inc"):
                            self.attribute_settings[idx]=0
                            repaired = True

        return self
                
    def calcFitness(self):
        # sim_list is a similarity dictionary (key=song1 / value=tuple(songX name, similariy value of songX with song1)
        sim_list = {}
        for index1, row1 in self.df1.iterrows():
                inner_dic = {}
                for index2, row2 in self.df2.iterrows():
                    # row1 = song1
                    # row2 = song2
                    r1 = row1.tolist()
                    r2 = row2.tolist()
                    
                    # filter out feature values which are not set in self.attribute_settings
                    rn1 = []
                    rn2 = []
                    for i in range(0,len(self.attribute_settings)):
                        if(self.attribute_settings[i]==1):
                            rn1.append(r1[i])
                            rn2.append(r2[i])
                    
                    # with filtered rows rn1 and rn2 perform similarity measurement
                    if(self.similarity=="euclid"):
                        value = np.linalg.norm(np.transpose(rn1)-np.transpose(rn2))
                    elif(self.similarity=="pearson"):
                        r,p = pearsonr((np.transpose(rn1))     ,     (np.transpose(rn2)))
                        if(r<0):
                            value = 1+r
                        else:
                            value = 1-r
                    else:
                        print("WRONG SIMILARITY OPTION")
                
                    inner_dic[index2] = value
                # sort list of songs from similar songs to different songs  
                sim_list[index1] = sorted(inner_dic.items(), key=operator.itemgetter(1))  
                
        
        #calculte the rank of each song and save it in ranks
        ranks = []
        for song in sim_list:
            for rank in range(0,len(sim_list[song])):
                if(sim_list[song][rank][0] == song):
                    ranks.append((song,rank))
        
        # calculate the mean rank using ranks and set self.fitness= mean rank
        summ = 0
        for song in ranks:
            summ += song[1]
        self.fitness = summ/len(ranks)
        return self.fitness


class Population:
# a Population is a crowd of "creatures" in the genetic algorithm concept
# It has following attributes:

# popsize = the number of individuals which "live" in this population

# population = a list of Individual objects which are part of this population

# nofattributes = the number of attributes the Individuals in this population have

# used_attributes = the number of attributes which are set to 1 in the Individuals attribute_settings

# mutprob = the mutation probability when two individuals of this population are crossed

# similarity = the similarity measure method used when calculating the fitness of the Individuals in this population

    def __init__(self,nofattributes,used_attributes,popsize,mutprob,similarity,population=[]):
        self.popsize = popsize
        self.population = population
        self.nofattributes = nofattributes
        self.used_attributes = used_attributes
        self.mutprob = mutprob
        self.similarity = similarity
    
    def add(self,ind):
        # adds an Individual to this population
        self.population.append(ind)
    
    def random_populate(self):
        # generates random Individuals and adds them to this population until the population size is reached
        self.population = []
        while(len(self.population)<self.popsize):
            self.add(Individual(self.nofattributes,self.used_attributes,self.similarity))
    
    def random_populate_multicore(self,cores):
        # generates random Individuals and adds them to this population until the population size is reached using multiple cores
        # cores specifies how many cores are used
        
        # start with an empty population 
        self.population = []
        
        # Queue acts as a thread-safe memory for created Individuals
        q = Queue()
        
        # determine chunk sizes for each thread
        chunks = [self.popsize // cores]*cores
        rest = self.popsize - sum(chunks)
        for i in range(0,cores-1):
            if rest>0:
                chunks[i]+=1
                rest-=1
        
        # create processes with corresponding chunk sizes
        processes = []
        for core in range(0,cores):
            processes.append(Process(target=self.createRandomChildren, args=(chunks[core],self.similarity,q)))
        
        # start processes
        for process in processes:
            process.start()
        
        # collect results
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
        
        # terminate processes
        for process in processes:
            process.join()
         
        # add all created children from results to the population
        for random_Individual in results:
            self.add(random_Individual)

            
    def createRandomChildren(self,nofchildren,similarity,q):
        # helper function for multicore functions
        # creates nofchildren random Individuals and adds them to the queue q
        for i in range(0,nofchildren):
            q.put(Individual(self.nofattributes,self.used_attributes,similarity))

        
    def getMeanFitness(self):
        # calculate the mean fitness of this populations Individuals
        summ = 0
        for ind in self.population:
            summ+=ind.fitness
        return summ/len(self.population)
    
    def getMaxFitness(self):
        # returns the Individual with the best fitness of this population
        best = self.population[0].fitness
        best_ind = self.population[0]
        for ind in self.population:
            if(ind.fitness<best):
                best = ind.fitness
                best_ind = ind 
        return(best_ind)
    
    def selectPartners(self):
        # select 2 partners from the population which later can be mated/crossed with cross()
        # To select partners a roulette wheel selection is performed depending on Individuals fitness
        # The greater the fitness the more likely it is, that an Individual is selected
        
        #result list
        partners = []
        # dictionary with key=Individual value=1/fitness
        # 1/fitness because a high mean rank is bad, whereas a low mean rank is considered a good fitness
        dic = {}
        for ind in self.population:
            dic[ind] = (1/ind.fitness)
        
        # sum of the fitness values = perimeter of the roulette wheel
        max = sum(dic.values())
        
        # spin the roulette wheel two times for selecting two partners
        for i in range(0,2):
            # spin wheel
            pick = np.random.uniform(0,max)
            
            # determine the Individual selected from the wheel
            current = 0
            for key, value in dic.items():
                current += value
                if current > pick:
                    # select Individual and append to result list
                    partners.append(key)
                    break
                    
        return partners
    
    def getNextGeneration(self):
        # generate the next generation out of this population
        # Indivudals are selected, crossed, mutated and the resulting children are appended to a new Population object
        
        # create new Population
        new_pop = Population(self.nofattributes,self.used_attributes,self.popsize,self.mutprob,self.similarity,population=[])
        
        # fill until popsize is reached
        while(len(new_pop.population) < new_pop.popsize):
            partners = self.selectPartners()
            child = partners[0].cross(partners[1],self.mutprob)
            new_pop.add(child)
        
        return new_pop

    def createChildren(self,nofchildren,q):
        # helper function for multicore functions
        # creates nofchildren Individuals by selection and crossing from this Population and adds them to the queue q
        for i in range(0,nofchildren):
            partners = self.selectPartners()
            child = partners[0].cross(partners[1],self.mutprob)
            q.put(child)

    def getNextGeneration_multicore(self,cores):
        # generate the next generation out of this population using multiple cores
        # Indivudals are selected, crossed, mutated and the resulting children are appended to a new Population object
        
        # create new Population
        new_pop = Population(self.nofattributes,self.used_attributes,self.popsize,self.mutprob,self.similarity,population=[])

        if True:
            # Queue acts as a thread-safe memory for created Individuals
            q = Queue()
            
             # determine chunk sizes for each thread
            chunks = [self.popsize // cores]*cores
            rest = self.popsize - sum(chunks)
            for i in range(0,cores-1):
                if rest>0:
                    chunks[i]+=1
                    rest-=1
            
            # create processes with corresponding chunk sizes
            processes = []
            for core in range(0,cores):
                processes.append(Process(target=self.createChildren, args=(chunks[core],q)))
            
            # start processes
            for process in processes:
                process.start()

            # collect results
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

            #terminate processes
            for process in processes:
                process.join()
            
            # add all created children from results to the population
            for child in results:
                new_pop.add(child)

            return new_pop

class experiment:
# an experiment is a easy to use framework class for performing a genetic algorithm based search
# It has following attributes:

# nofcores = number of cores used for calcuations (population initialization and calculation of new generations)

# popsize = the population size of the generations

# used_attributes=  the number of attributes which are set in each Individuals attribute_settings

# iterations = how many generations are created/searched by the genetic algorithm

# mutprob = the mutation probabilty while crossing

# similarity = the similarity measure used for calculating the fitness of Individuals

# nofattributes = the length of each Individuals attribute_settings (42 features is default because 42 extracted features)

    def __init__(self, nofcores,popsize,used_attributes,iterations,mutprob,similarity,nofattributes=42):
        self.nofcores = nofcores
        self.popsize = popsize
        self.used_attributes = used_attributes
        self.iterations = iterations
        self.mean_fitness = []
        self.alltime_best_fitness = []
        self.current_best_fitness = []
        self.nofattributes = nofattributes
        self.mutprob = mutprob
        self.similarity = similarity

    def start(self):
        # create new population
        p = Population(self.nofattributes,self.used_attributes,self.popsize,self.mutprob,self.similarity)
        start_time = time.time()
        # populate population with random Individuals
        p.random_populate_multicore(self.nofcores)
        print("rand init Time was: "+str(round((time.time() - start_time))))
        
        # calculate and analyse the next generation
        for iteration in range(0,self.iterations):
            if(iteration%10==0):
                print("iteration: "+str(iteration))
                
            #save mean fitness
            self.mean_fitness.append(p.getMeanFitness())
            #save individual with best fitness in this population
            current_fitness = p.getMaxFitness()
            self.current_best_fitness.append((current_fitness,current_fitness.fitness))
            #save alltime best individual since experiment started
            if(iteration==0 or current_fitness.fitness < self.alltime_best_fitness[-1][1]):
                self.alltime_best_fitness.append((current_fitness,current_fitness.fitness))
            else:
                self.alltime_best_fitness.append(self.alltime_best_fitness[-1])
            # calculate the next generation multicore
            p = p.getNextGeneration_multicore(self.nofcores)
        
        #save the final last population after all iterations finished
        self.finalpop = p
        
    def plot(self):
        # plot the results of this experiment
        alltime = [x[1] for x in self.alltime_best_fitness]
        curr = [x[1] for x in self.current_best_fitness]

        plt.plot(self.mean_fitness)
        plt.plot(curr)
        plt.plot(alltime)
        plt.ylabel('mean rank')
        plt.xlabel('GA iterations')
        plt.legend(['pop mean fitness', 'current pop best', 'alltime best'], loc='upper left')
        plt.show()