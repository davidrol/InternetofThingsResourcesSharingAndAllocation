
#!/usr/bin/python
import numpy as np
import math
from random import uniform
# from scipy.spatial import distance

# return array dimension
def matrixLenght(matrice):
    return matrice.shape  #(3, 2) -> 3 rows and 2 columns for example

# Summation of List
def listsum(numList):
    theSum = 0
    for i in numList:
        theSum = theSum + i
    return theSum

print(listsum([1,3,5,7,9]))

# Recursive summation of List
def listrecsum(numList):
   if len(numList) == 1:
        return numList[0]
   else:
        return numList[0] + listrecsum(numList[1:])
print(listrecsum([1,3,5,7,9]))


# ::::::::::::::::    calcul distance pour une ressource donnée : Lits, Masques, respirateurs, etc
# ::::::::::::::::    ie ligne par ligne
# ::::::::::::::::    le resultat est une matrice de distance de même taille que les individus d'entrée.
# ::::::::::::::::    calcul de l'ecart entre un individu (solution proposée)et la demande exprimée : distance pour une ressource donnée : Lits, Masques, respirateurs, etc

class individu:
    def __init__ (self,nombre_ressources=2,nombre_services=2):      # Deux services et deux ressources par défauts
        #   self.NB_genes       = nombre_genes                      # Nombre de gènes de la population
        self.NB_ressources  = nombre_ressources                     # Nombre de catégories de ressources
        self.NB_services    = nombre_services                       # Nombre de services connectés dans l'espace de partage collaboratif
        #self.distances     = [[]] * nombre_genes                   # Matrice des distances
        self.solution       = [[]]                                  # Matrice solution d'allocation de ressources de l'individu
        self.solution       = np.random.randint(30,150,size=(nombre_ressources,nombre_services))

        #self.NB_initial_population = NB_initial_population  # Taille de la populatio initiale
        # for ind in range(self.NB_initial_population):
        #     self.individus.append (individu(NB_genes))

    def ecart(self, demandes):
        # Mutation changes a single gene in each offspring randomly.
        matdim = matrixLenght(self.solution)
        nbligne     = matdim[0] # Number rows
        nbcols      = matdim[1] # Number colums
        # distances =   [q1,q2,q3,q4,q5,q6,q7] : il y'a 7 ressources
        distances   = np.random.randint(1,10,size=(nbligne))
        #print(distances)
        # Parcours des lignes ressource par ressource pour calculer la distance entre les allocations et la demande exprimée.
        for idy in range(nbligne):
            qi = np.linalg.norm(demandes[idy]-self.solution[idy])
            #qi = np.dist(demandes[idy],individu1[idy]) # equivalent to sqrt(sum((px - qx) ** 2.0 for px, qx in zip(p, q)))
            #for idx in range(nbcols): # Parcours des colones
                #calcule de  qi=SUM(SQRT((di - ri)^2))
                #qi+ = math.sqrt(demandes[idy,idx] - individu1[idy,idx])^2) # revoir cette ligne avec les fonctions math
            distances [idy] = qi # affectation de l'écart dans le vecteurs distances
            qi = 0 # reinitialisation de la distance
        return distances

    # ::::::::::::::::: calcul de la fonction objectif (fitness): f = SUM (wjxqj) / SUM (wj)
    def fitnessFunction(self, ecarts, weights): # f = SUM(wj*qj) / SUM (wj) minimiser f
        # On suppose qu'on a calculé tout les qi pour chaque ressource à allouer.
        # On a donc mis dans la matrice distances tous les qi
        # On considère que tous les poids ie des priorités sur les ressources sont dans la matrice weights
        # ecarts = [q1,q2,q3,q4,q5,q6,q7] : il y'a 7 ressources
        # weights =   [w1,w2,w3,w4,w5,w6,w7] : il y'a 7 poids (priorités)
        matdim = matrixLenght(ecarts)
        #nbligne     = matdim[0] # Number rows
        nbcols      = matdim[0] # Number cols
        objectiveFunction = 0
        sum_Weights = 0
        somme   = 0
        #print(nbcols)
        sum_Weights =  np.sum(weights) # return the sum of weights in the list
        #for idy in range(nbcols): # Parcours des lignes
            #sum_Weights+ =  weights[idy] # SUM(wj)
        for idy in range(nbcols): # Parcours des lignes
            # The random value to be added to the gene.
            # print("Shape test : "+ str(distances))
            somme = somme + weights[idy] * ecarts[idy] # SUM(wj*qj)
            somme =  somme / sum_Weights # SUM(wj*qj) / SUM (wj)
        return somme

# ::::::::::::::::: Calcul du meilleur individu entre deux individus
# ::::::::::::::::: on prend en entré deux individu et calcule les distance euclidienne entre les deux individus ressource par ressources,
# ::::::::::::::::: ie ligne par ligne
# ::::::::::::::::: le resultat est une matrice de distance de même taille que les individus d'entrée.
# ::::::::::::::::: L'individu 1 est meilleur que l'individu 2
# ::::::::::::::::: lorsque la fonction de fitness (distance euclid.) entre ind1 et la demande est meilleur que ind2 et la demande .

    def compare(self,individu2, demandes, weights):
        matdim  = matrixLenght(individu2.solution) # We suppose that indiv1 and indiv2 have the same dimensions
        nbligne = matdim[0] # Number of rows
        nbcols  = matdim[1] # Number of colums
        # distances =   [q1,q2,q3,q4,q5,q6,q7] : il y'a 7 ressources
        distancesInd1   = self.ecart(demandes)
        distancesInd2   = individu2.ecart(demandes)
        #print("Ecarts individu 1 : "+ str(distancesInd1))
        #print("Ecarts  individu 2 : "+ str(distancesInd2))
        # Calcule de la fonction objective pour individu 1 et pour individu2
        objectiveFunctInd1  = self.fitnessFunction(distancesInd1, weights)
        objectiveFunctInd2  = individu2.fitnessFunction(distancesInd2, weights)
        meilleurInd = min(objectiveFunctInd1, objectiveFunctInd2) # meilleur fonction objective
        #if meilleurInd  == objectiveFunctInd1 :
        #    return self.solution
        #else: return individu2.solution # on retourne la meilleur fonction objective
        if meilleurInd  == objectiveFunctInd1 :
            return self
        else: return individu2 # on retourne la meilleur fonction objective
#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#::::::::::::::::::::::::::::::
#:::::::::::::::::::::          resources_random_alocation : distribue des ressources aleatoirement aux services conectés :                  #:::::::::::::::::::::
#:::::::::::::::::::::              : on vérifie que la distribution ne dépasse pas la quantité disponible pour chaque catégorie disponibles #:::::::::::::::::::::
#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#::::::::::::::::::::::::::::::
    def resources_random_alocation (self, resources_available, demandes):
        #1-récuperer l'ensemble des ressources disponibles dans self.available_resources : matrice à deux dimensions
        #2-pour chaque resource (cétégorie), calculer la somme sur l'ensemble des services conectés
        #3-definir une lois de distribution
        #4-distribuer les resources à chaque service de sorte à ne pas dépasser la quantité totale disponible
        total_per_categories = []
        #extraction d'une catégorie de ressources eg: lits / casques / respirateur / une ligne de la matrice
        # self.available_resources # Chaque services connectés à déclaré sa quantité de ressources disponibles
        for categorie in range(self.NB_ressources) :
            total_res =  np.sum(resources_available[categorie]) # totale des resources disponibles pour une catégorie
            total_per_categories.append(total_res)
        # distribution aleatoire des ressources aux servcies dans la matrice solution de l'individu.
        # nous faison sune partage ressource par ressource, ie ligne par ligne
        i = 0 # curseur sur les services pour extraire les demnades formulés
        for categorie in range(self.NB_ressources) :
            total_cat = total_per_categories[categorie] # totale des resources disponibles pour une catégorie
            nb_services = self.NB_services # nombre de services conectés demandeurs de ressources
            # il ne faut pas alloué plus de ressources que le service en a demandés
            # car on se place dans le contexte ou la ressource est critique donc inutile d'alloué par exemple 1à respirateurs à un service qui en a demandé que 4
            demande_service_i = demandes[categorie][i] # contient la demande formulé par ce service
            #génération d'un nombre aleatoire entre [1,min(total_cat,demande_service_i)]
            part = np.random.randint(1,min(total_cat,demande_service_i))  # part de ressource alloué à ce service
            #affectation de la par à l'individu en question dans la matrice des individus
            self.solution[categorie][i] = part
        return self

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#::::::::::::::::::::: Define population Class and methods                                  :::::::::::::::::::::::::::::::::::::::::::::
#:::::::::::::::::::::      Class Population                                                :::::::::::::::::::::::::::::::::::::::::::::
#:::::::::::::::::::::              def __init__ (self, NB_genes,NB_initial_population)     :::::::::::::::::::::::::::::::::::::::::::::
#:::::::::::::::::::::              def selection (self, NB_genes,NB_initial_population)    :::::::::::::::::::::::::::::::::::::::::::::
#:::::::::::::::::::::              def crossover (self, NB_genes,NB_initial_population)    :::::::::::::::::::::::::::::::::::::::::::::
#:::::::::::::::::::::              def crossover_two (self, indiv1,indiv2)                 :::::::::::::::::::::::::::::::::::::::::::::
#:::::::::::::::::::::              def crossover_all (self)                                :::::::::::::::::::::::::::::::::::::::::::::
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
class population:
    def __init__ (self,nombre_ressources,nombre_services,resource_available,distances,priority,demande_init,nb_initial_population=60):
        #self.NB_genes       = nombre_genes                          # Nombre de gènes de la population
        self.NB_ressources  = nombre_ressources                     # Nombre de catégories de ressources
        self.NB_services    = nombre_services                       # Nombre de services connectés dans l'espace de partage collaboratif
        self.distances      = distances                             # [[]] Matrice des distances entre les services IoTS connectés
        self.weights        = priority                              # [[]] Matrice des poids, priorités entre les ressources partagées
        self.individus      = []                                    # Matrice des individus de la population
        self.NB_initial_population  = nb_initial_population  # Taille de la populatio initiale
        self.available_resources    = resource_available
        self.demandes = demande_init
        # à revoir l'initialisation des individus
        for i in range(self.NB_initial_population):
            # self.individus.append(individu(self.NB_ressources,self.NB_services))
            ind = individu(self.NB_ressources,self.NB_services)
            ind.resources_random_alocation (self.available_resources, self.demandes) # On initialise les individus avec les ressources disponibles
            self.individus.append (ind)
#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::
#:::::::::::::::::::::          Croisement : croise les gênes de deux individus pour en creer nouveaux troisième        #:::::::::::
#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::
    def crossover_two (self, indiv1,indiv2):
        resultatCrossover = []  # Le résultat du crosOver retourne deux individus
        indivResult1 = individu(self.NB_ressources,self.NB_services) # Les individus de la population ont la même taille
        indivResult2 = individu(self.NB_ressources,self.NB_services)
        # calcul des tailles des matrices des individus
        # matdim  = matrixLenght(individu1) # We suppose that indiv1 and indiv2 have the same dimensions
        # nbligne = matdim[0] # Number of rows
        # nbcols  = matdim[1] # Number of colums
        sequence1, sequence2    = np.hsplit(indiv1.solution,[3]) # On divise le gene de l'individu 1 en deux parties
        sequence3, sequence4    = np.hsplit(indiv2.solution,[3]) # On divise le gene de l'individu 2 en deux parties
        #print("sequences des individus Seq1: ")
        #print(str(sequence1))
        #print("sequences des individus Seq4: ")
        #print(str(sequence4))
        indivResult1.solution   = np.concatenate((sequence1,sequence4),1) # On concatene seq1 et seq4 des deux genes issues des individus 1 et 2 pour former le troisième individu.
        indivResult2.solution   = np.concatenate((sequence3,sequence2),1) # On concatene seq2 et seq3 des deux genes issues des individus 1 et 2 pour former le quatrième individu.
        # resultatCrossover.append(str(x))
        resultatCrossover.append(indivResult1)
        resultatCrossover.append(indivResult2)
        return resultatCrossover # Retourne deux individus


#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::
#:::::::::::::::::::::          Croisement : croise les gênes de toute la population des individus pour remplacer les individus par de nouveaux#:::::::::::
#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::
    def crossover_population (self):
        individus_pop = self.individus
        # nb_individus  = np.size(individus_pop) # We suppose that indiv1 and indiv2 have the same dimensions
        # for index in range (nb_individus):
        #     print("Test cross_over_pop:  "+"index indiv : "+str(index)+str((self.individus[index]).solution))
        nb_individus  = np.size(individus_pop) # We suppose that indiv1 and indiv2 have the same dimensions
        # Number of rows
        # nbcols  = matdim[1] # Number of colums
        index_middle = math.floor(nb_individus/2) # index de l'individu au milieu take the smallest integer
        # On croise les individus deux par deux en les remplaçant dans la population initiale.
        # print("Taille de la population:  "+str(nb_individus))
        # print("Index Milieu:  "+str(index_middle))
        # print("Test cross_over_population:  "+str(self.individus))
        i = 0
        j = nb_individus-1
        for index in range (index_middle):
            indiv1 = individu(self.NB_ressources,self.NB_services) # Les individus de la population ont la même taille
            indiv2 = individu(self.NB_ressources,self.NB_services)
            indiv1.solution = self.individus[i].solution
            indiv2.solution = self.individus[j].solution
            # print("i:  "+str(i))
            # print("j:  "+str(j))
            # print("Test cross_over_population: indiv1.solution  "+str(indiv1.solution))
            # print("Test cross_over_population: indiv2.solution  "+str(indiv2.solution))
            sequence1, sequence2    = np.hsplit(indiv1.solution,[3]) # On divise le gene de l'individu 1 en deux parties
            sequence3, sequence4    = np.hsplit(indiv2.solution,[3]) # On divise le gene de l'individu 2 en deux parties
            indiv1.solution   = np.concatenate((sequence1,sequence4),1) # On concatene seq1 et seq4 des deux genes issues des individus 1 et 2 pour former le troisième individu.
            indiv2.solution   = np.concatenate((sequence3,sequence2),1) # On concatene seq2 et seq3 des deux genes issues des individus 1 et 2 pour former le quatrième individu.
            i = i+1
            j = j-1
        return self.individus # Retourne les individus de la population

#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::
#:::::::::::::::::::::          mutation : permet de changer les genes d'un individu aléatoirement                      #:::::::::::
#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::
    def mutation (self,individu):
        #resultat = {}
        #indivResult = individu(self.NB_ressources,self.NB_services) # Les individus de la population ont la même taille
        # calcul des tailles des matrices des individus
        matdim  = matrixLenght(individu.solution) # We suppose that indiv1 and indiv2 have the same dimensions
        nbligne = matdim[0] # Number of rows
        nbcols  = matdim[1] # Number of colums
        for res in range (nbligne):
            (individu.solution[res])[::-1] # reverse the resource allocation row by row
        return individu

# revoir cette fonction array index out of bound

#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::
#:::::::::::::::::::::          mutation : permet de changer les genes d'un individu aléatoirement                      #:::::::::::
#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::
    def mutation_population (self):
        #resultat = {}
        #indivResult = individu(self.NB_ressources,self.NB_services) # Les individus de la population ont la même taille
        # calcul des tailles des matrices des individus
        # matdim  = matrixLenght(individu1) # We suppose that indiv1 and indiv2 have the same dimensions
        # nbligne = matdim[0] # Number of rows
        # nbcols  = matdim[1] # Number of colums
        index = 0
        #np.size(individus_pop)
        for index in range (np.size(self.individus)):
            indiv = self.individus[index]
            mutate = self.mutation(indiv)
            self.individus[index] = mutate
            index = index+1
        return self.individus

#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#::::::::::::::
#:::::::::::::::::::::          Calcule la fonction de fitness pour tous les individus d'une population.                    ::::::#::::::::::::::::
#:::::::::::::::::::::          il s'agit d'évaluer tout les individus de la Population et retourner un vecteur evaluation. ::::::#::::::::::::::::
#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#::::::::::::::
    def fitness_population (self):
        result_fitness = []
        #fitnessFunction(distances, weights):
        i=0
        for ind in self.individus:
            result_fitness.append (ind.fitnessFunction(ind.ecart(self.demandes), self.weights))
            i = i + 1
        return result_fitness
        #def ecart(self, demandes):
        #def fitnessFunction(self, ecarts, weights)

#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#::::::::::::::::
#:::::::::::::::::::::          selection : Selecting the best individuals in the current generation as parents for the next generation. #:::::::::::
#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#::::::::::::::::
    def selection (self,nbselection): # donner la possibilité que la fonction soit utilisée sur d'autres populations
        # NB: si la fonction ne marhe pas, on poura indexer les éléments du dictionaire population.individus
        selected_parents    = []                    # liste des individus résultat de la sélection
        pop_fitness = self.fitness_population()     # retourne le vecteur évaluation des individus de la population
        sorted_pop_fitness = pop_fitness   # sort the fitness array by values to have an ordered list of fitness values
        sorted_pop_fitness = np.sort(sorted_pop_fitness)[::-1]#(descending order)   # reverse the sort to have max on the top of the list : sort the fitness array by values to have an ordered list of fitness values
        #print("Test selection  -> pop_fitness:  "+str(pop_fitness))
        #print("Test selection  -> sorted_pop_fitness:  "+str(sorted_pop_fitness))
        # 1-evaluation de tous les individus
        # 2-selection des meileurs en fonction du fitness de chaque individu
        # 3-insertion dans la liste résultat
        i = 0
        for ind in range(nbselection): # Selectionner le nombre d'individus demandé
            # select a value in the fitness sorted list
            value_max_sorted = sorted_pop_fitness [i]   # Onrecupère la valeur du meilleur dans la liste triée.
            # print("Test selection  -> value_max_sorted:  "+str(value_max_sorted))
            index_indiv  = np.where(pop_fitness == value_max_sorted)[0].tolist()   # on récupere l'indice du meilleur retourne un vecteur d'une sele case array[0]
            # print("Test selection  -> numpy.where(): index_indiv "+str(index_indiv))
            #selected_parents.append(self.individus[i])
            index_max = index_indiv[0]
            # print("Test selection  -> index_max: "+str(index_max))
            indiv_max = self.individus[index_max]
            # print("Test selection  -> indiv_max: "+str(indiv_max))
            # max_fitness = self.individus[]     # on sait que le meilleur est dans la liste des individus de a population
            selected_parents.append(indiv_max)   # ajout du meilleur dans la liste des futurs parents
            i = i + 1                            # incrémentation pour atteindre le nombre d'indivudus à selectionner
        return selected_parents                  # ne contient que les refs sur les individus: pour acceder aux solutions, il faut faire appel à sol= individu.solution selected_parents[i].solution

#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#::::::::::::::::
#:::::::::::::::::::::          IoT  Smart Ressource Sharing for collaboration and allocation optimisation                          #::::::::::::::::
#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#::::::::::::::::
    class IoTS_ressource_sharing_allocator:

#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#::::::::::::::::
#:::::::::::::::::::::          iots_ressources_defnition : Define ressources in an IoTS  network                                        #:::::::::::
#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#::::::::::::::::
        def iots_ressources_network_defnition(population, resources):
            res_def =0
            return res_def
#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#::::::::::::::::
#:::::::::::::::::::::          iots_ressources_defnition : Define ressources in an IoTS  network                                        #:::::::::::
#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#::::::::::::::::
        def iots_ressources_occurence_definition(population, resources):
            res_occ = 0
            return res_def
#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#::::::::::::::::
#:::::::::::::::::::::          iots_ressources_defnition : Define ressources in an IoTS  network                                        #:::::::::::
#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#::::::::::::::::
        def iots_ressources_availability_definition(population, resources):
            res_available =0
            return res_available
#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#::::::::::::::::
#:::::::::::::::::::::          iots_ressources_defnition : Define ressources in an IoTS  network                                        #:::::::::::
#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#::::::::::::::::
        def iots_ressources_demands_definition(population, resources):
            res_demands = 0
            return res_demands
#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#::::::::::::::::
#:::::::::::::::::::::          iots_ressources_defnition : Define ressources in an IoTS  network                                        #:::::::::::
#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#::::::::::::::::
        def iots_services_network_definition(population, resources):
            res_net = 0
            return res_net
#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#::::::::::::::::
#:::::::::::::::::::::          iots_ressources_priority_definition : Define ressources in an IoTS  network                              #:::::::::::
#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#::::::::::::::::
        def iots_ressources_priority_definition(population, resources):
            res_prior =0
            return res_prior
#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#::::::::::::::::
#:::::::::::::::::::::          iots_network_services_distances_definition : Define distances between services in an IoTS network        #:::::::::::
#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#::::::::::::::::
        def iots_network_services_distances_definition(population, services, distance_matrice):
            dist_matrice =0
            return dist_matrice
#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#::::::::::::::::
#:::::::::::::::::::::          iots_network_ressources_constraints_definition : Define the constraints on ressources in an IoTS network #:::::::::::
#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#::::::::::::::::
        def iots_network_ressources_constraints_definition(population, services, constraints):
            dist_matrice =0
            return dist_matrice
#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#::::::::::::::::
#:::::::::::::::::::::          iots_ressource_Allocator : optimaly allocate ressources in real time i a collaborative IoT platform      #:::::::::::
#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#:::::::::::::::::::::#::::::::::::::::
        def iots_ressources_allocation_optimisation (popupaltion,nbselection): # donner la possibilité que la fonction soit utilisée sur d'autres populations
            optimal_allocation = 0
            return optimal_allocation




















#::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# def select_mating_pool(pop, fitness, num_parents):
#     # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
#     parents = numpy.empty((num_parents, pop.shape[1]))
#     for parent_num in range(num_parents):
#         max_fitness_idx = numpy.where(fitness == numpy.max(fitness))
#         max_fitness_idx = max_fitness_idx[0][0]
#         parents[parent_num, :] = pop[max_fitness_idx, :]
#         fitness[max_fitness_idx] = -99999999999
#     return parents
#
#























#;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


##### ::::::::::::::::::::::::::
#
# class population:
# 	def __init__ (self,nombre_genes,nombre_initial_population=50):
# 		self.nombre_genes = nombre_genes
# 		self.distances    = [[]] * nombre_genes
# 		self.individus    = []
# 		self.nombre_initial_population = nombre_initial_population
# 		for v in range(self.nombre_initial_population):
# 			self.individus.append (individu(nombre_genes))
# 	"""
# 	selection : methode de selection des meilleurs N individus
# 	"""
# 	def selection (self , N=-1):
# 		if N == -1 : N = len(self.individus)
#
# 		for v in self.individus:
# 			v.evalutation (self.individus,self.distances)
#
# 		self.individus.sort (lambda a,b : a.valeur_individu - b.valeur_individu)
# 		if N <= len(self.individus) : self.individus = self.individus[0:N]
# 	"""
# 	croisement : selectionne  deux individus et croise leur gênes pour en creer de nouveaux
# 	"""
# 	def croisement_deux (self , i1 , i2):
# 		i = individu (self.nombre_genes)
# 		l = len(i1.genes)
# 		a = l / 2
# 		p2 = i2.genes[a:l]
# 		p1 = i1.genes[0:a]
# 		choices = range(0,l)
# 		for v in p1:
# 			try:choices.remove (v)
# 			except : pass
# 		for v in p2:
# 			try : choices.remove (v)
# 			except : pass
# 		idc = 0
# 		for  id in range(len(p2)):
# 			if p2[id] in p1:
# 				p2[id] = choices[idc]
# 				idc = idc + 1
# 		i.genes = p1 + p2
# 		return i
# 	""" croise tous les individus entre eux"""
# 	def croisement_all (self):
# 		new = []
# 		for v1 in range(len(self.individus)):
# 			for v2 in  range(v1,len(self.individus)):
# 				new.append ( self.croisement_deux (self.individus[v1] , self.individus[v2]))
# 		self.individus  = self.individus + new
#
# 	""" croise nombre_croisement individus aleatoires entre eux """
# 	def croisement_nombre (self , nombre_croisement):
# 		new = []
# 		for r in range(nombre_croisement):
# 			v1 = random.randint (0 , len(self.individus)-1)
# 			v2 = random.randint (0 , len(self.individus)-1)
# 			new.append (self.croisement_deux (self.individus[v1] , self.individus[v2]) )
# 		self.individus = self.individus + new
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#
# def cal_pop_fitness(equation_inputs, pop):
#     # Calculating the fitness value of each solution in the current population.
#     # The fitness function caulcuates the sum of products between each input and its corresponding weight.
#     fitness = numpy.sum(pop*equation_inputs, axis=1)
#     return fitness
#
# def select_mating_pool(pop, fitness, num_parents):
#     # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
#     parents = numpy.empty((num_parents, pop.shape[1]))
#     for parent_num in range(num_parents):
#         max_fitness_idx = numpy.where(fitness == numpy.max(fitness))
#         max_fitness_idx = max_fitness_idx[0][0]
#         parents[parent_num, :] = pop[max_fitness_idx, :]
#         fitness[max_fitness_idx] = -99999999999
#     return parents
#
# def crossover(parents, offspring_size):
#     offspring = numpy.empty(offspring_size)
#     # The point at which crossover takes place between two parents. Usually it is at the center.
#     crossover_point = numpy.uint8(offspring_size[1]/2)
#
#     for k in range(offspring_size[0]):
#         # Index of the first parent to mate.
#         parent1_idx = k%parents.shape[0]
#         # Index of the second parent to mate.
#         parent2_idx = (k+1)%parents.shape[0]
#         # The new offspring will have its first half of its genes taken from the first parent.
#         offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
#         # The new offspring will have its second half of its genes taken from the second parent.
#         offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
#     return offspring
#
# def mutation(offspring_crossover):
#     # Mutation changes a single gene in each offspring randomly.
#     for idx in range(offspring_crossover.shape[0]):
#         # The random value to be added to the gene.
#         random_value = numpy.random.uniform(-1.0, 1.0, 1)
#         offspring_crossover[idx, 4] = offspring_crossover[idx, 4] + random_value
#     return offspring_crossover
