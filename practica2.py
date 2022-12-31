# -*- coding: utf-8 -*-

# -- Sheet --

# # Práctica 2: Algoritmos metaheurísticos
# 
# ## Sistemas Inteligentes
# 
# ### Curso académico 2022-2023
# 
# #### Profesorado:
# 
# * Juan Carlos Alfaro Jiménez (`JuanCarlos.Alfaro@uclm.es`)
# * Guillermo Tomás Fernández Martín (`Guillermo.Fernandez@uclm.es`)
# * María Julia Flores Gallego (`Julia.Flores@uclm.es`)
# * Ismael García Varea (`Ismael.Garcia@uclm.es`)
# * Luis González Naharro (`Luis.GNaharro@uclm.es`)
# * Aurora Macías Ojeda (`Profesor.AMacias@uclm.es`)
# * Marina Sokolova Sokolova (`Marina.Sokolova@uclm.es`)


# ## 0. Preliminares
# 
# Antes de comenzar con el desarrollo de esta práctica es necesario **descargar**, en **formato `.py`**, el código de la **práctica anterior** con el nombre **`utils.py`**. Para ello, pulsamos, en la libreta de la primera práctica, **`File > Export .py`**.
# 
# Una vez hemos descargado y nombrado correctamente el fichero, lo **añadimos** al espacio de trabajo de la **libreta** de la práctica **actual** a través de **`Attached data > Notebook files > Upload files`** y subimos el fichero `utils.py` descargado en el paso anterior.
# 
# Tras esto, debemos **cambiar** el **constructor** de la clase **`Problem`** para que **reciba** directamente el **problema** a resolver, **en lugar de cargarlo desde** un **fichero**. Esto se debe a que los algoritmos metaheurísticos van a tener que resolver, en múltiples ocasiones, este problema. De esta manera nos **evitamos** la **carga computacional extra** que implica **leer** el problema desde un **fichero**. Además, también es necesario **comentar** cualquier línea de **código** que **imprima estadísticas** para evitar salidas largas.


# ## 1. Introducción
# 
# En esta práctica, vamos a **resolver** un **problema** de **optimización combinatoria** mediante **algoritmos metaheurísticos**. En particular, vamos a implementar **algoritmos genéticos** para abordar el **problema** de **ruteo** de **vehículos**. En este, **varios vehículos** con **capacidad limitada** deben **recoger paquetes** en **diferentes ubicaciones** y **trasladarlos** a una **sede central** de recogida.
# 
# Además, se **analizará** y **comparará** el **rendimiento** de **diferentes algoritmos genéticos** (mediante la modificación de los pasos correspondientes) en diferentes instancias del problema.
# 
# ---


# ## 2. Descripción del problema
# 
# El concepto de mapa que vamos a utilizar es similar al de la primera práctica. Este se representa mediante un **grafo**, donde los **nodos** representan **ciudades** y los **enlaces** indican la existencia de una **carretera en ambos sentidos** entre dos ciudades. Además, los **enlaces** tienen un **peso** asociado indicando la **distancia real** entre las dos ciudades. Al mismo tiempo, se proporciona una **sede central**, que se trata de la ciudad donde se deben dejar los paquetes. A su vez, se dispone de una **flota de vehículos** con **capacidad limitada** que deben **recoger** los **paquetes** en las ciudades correspondientes y que **inicialmente** están aparcados en la **sede central**. **En caso de que a la hora de recoger un paquete se supere la capacidad del vehículo correspondiente, este debe volver a la sede central a descargar todos los paquetes, considerando el coste que implicaría volver**.
# 
# Un mapa es un problema en particular, pero diferentes paquetes, capacidades de vehículos y ubicación de la sede central pueden dar lugar a diferentes instancias del problema. Por tanto, el **objetivo** en este problema es **recoger todos los paquetes de tal manera que los vehículos recorran la menor distancia posible**.
# 
# **Con el objetivo de simplificar la práctica en evaluación continua, se asume que se cuenta con un único vehículo. No obstante, esto podría cambiar para la evaluación no continua.**
# 
# ---


# ## 3. Desarrollo de la práctica
# 
# Durante el desarrollo de la práctica se va a proporcionar un conjunto de mapas, sobre los cuáles se debe resolver el problema de optimización combinatoria correspondiente. Es importante destacar que la **dimensionalidad** del **problema** (número de ciudades, carreteras y paquetes) puede ser **variable**, por lo que los diferentes **algoritmos genéticos** deben ser lo suficientemente **eficientes** para que puedan **resolver** los **problemas** en un **tiempo razonable**.
# 
# **Además, algunos escenarios se van a guardar para las entrevistas de prácticas, por lo que el código debe ser lo más general posible para cargarlos de manera rápida y sencilla.**


# ### 3.1. Entrada
# 
# Cada escenario tendrá un fichero `.json` asociado con la siguiente estructura:
# 
# ```JSON
# {
#     "map": {
#         "cities": [
#             {
#                 "id": id_city_0,
#                 "name": name_city_0,
#                 "lat": latitude_city_0,
#                 "lon": longitude_city_0
#             }
#         ],
#         "roads": [
#             {
#                 "origin": origin_city_id,
#                 "destination": destination_city_id,
#                 "distance": road_distance
#             }
#         ]
#     },
#     "warehouse": warehouse_city_id,
#     "vehicles": [
#         {
#             "id": id_vehicle_0,
#             "capacity": capacity_vehicle_0
#         }
#     ]
#     "parcels": [
#         {
#             "id": id_parcel_0,
#             "city": parcel_city_id,
#             "weight": weight_parcel_0
#         }
#     ]
# }
# ```
# 
# Hay cuatro elementos principales en el fichero:
# 
# * `map`: Un diccionario con el mapa, cuya descripción es la misma que la de la primera práctica
# * `warehouse`: Identificador de la ciudad donde se encuentra la sede central
# * `vehicles`: Lista de vehículos disponibles
# * `parcels`: Lista de paquetes a recoger
# 
# Por su parte, `vehicles` contiene:
# 
# * `id`: Identificador del vehículo
# * `capacity`: Capacidad máxima del vehículo
# 
# Y `parcels`:
# 
# * `id`: Identificador del paquete
# * `city`: Ciudad donde se encuentra el paquete
# * `weight`: Peso del paquete
# 
# **Para añadir los ficheros con los problemas al espacio de trabajo se debe usar el mismo procedimiento anterior.**
# 
# ---


# ## 4. Plan de trabajo


# ### 4.1. Formalización del problema
# 
# Para resolver cualquier problema de optimización combinatoria en primer lugar hay que definir como vamos a **codificar** las **soluciones** al problema. Si bien es algo que se deja a criterio propio, se plantea la siguiente pregunta, **¿cuál puede ser la mejor representación para una secuencia de paquetes a recoger?**
# 
# Se puede comprobar si la respuesta es correcta introduciéndola en la variable `answer` del siguiente fragmento de código:


# Third party
import hashlib

check_answer = lambda answer, hashed: "The answer is " + ("" if hashlib.md5(answer).hexdigest() == hashed else "in") + "correct."

# TODO: Introduce here the answer to use for the hashing
answer = "Permutation" #La permutacion es la que hay que usar

# Avoid case sensitivity in the answer
answer = str.lower(answer)

# Encode the answer before hashing
answer = answer.encode("utf-8")

hashed = "90d377b31e1ac26d0d10d5612ce33ccc"  # The hashed answer
print(hashed)

check_answer(answer, hashed)

# ### 4.2. Implementación
# 
# A continuación se proporciona la estructura de clases recomendada para resolver el problema en cuestión. Tendréis que completar las siguientes clases de acuerdo con los algoritmos estudiados en teoría. **Debéis incluir en la siguiente celda todas las librerías que vayáis a utilizar para mantener la libreta lo más organizada posible**:


# TODO: Import here the libraries
import random
import json
import time
from utils import Problem, HeuristicaGeodesica, AStar

# #### Clase `Individual`
# 
# Esta clase proporciona la **codificación** de un **individuo** de la **población**.
# 
# Los **métodos obligatorios** que se deben añadir son:
# 
# * ``__init__(self, num_genes, generation_type, crossover_type, mutation_type)``: Inicializa el **número** de **genes** del **individuo** y el **tipo** de **operación** de **generación**, **cruce** y **mutación**. Ademas, genera la **solución** que **representa** el **individuo**.
# * ``generate(num_genes, generation_type)``: Método estático para **generar** una **solución** del tamaño proporcionado de acuerdo con el tipo de operación de generación.
# * ``crossover(self, individual)``: **Cruza** el **individuo actual** con el **individuo** de **entrada** de acuerdo con el tipo de operación de cruce.
# * ``mutation(self)``: **Muta** el **individuo** de acuerdo con el tipo de operación de mutación.
# * ``evaluate(self, problem)``: **Evalua** el **individuo** usando el **problema** a **resolver**.
# 
# Y los **métodos recomendados** son:
# 
# * ``__str__(self)``: **Representación** en formato de **cadena** de **caracteres** de un **individuo**. Método útil para depurar una lista de individuos.
# * ``__repr__(self)``: **Método** invocado cuando se ejecuta **``print``** sobre el **individuo**. Método útil para depurar un solo individuo.


class Individual:
    #El individuo se representa por la lista sucesiva de los id de los de paquetes recogidos

    # =============================================================================
    # Constructor
    # =============================================================================
    # TODO: Code here the constructor
    def __init__(self, num_genes, generation_type, crossover_type, mutation_type): 
        self.num_genes = num_genes
        self.generation_type = generation_type
        self.crossover_type = crossover_type
        self.mutation_type = mutation_type
        self.fitness = 0
        self.solution = Individual.generate(num_genes,generation_type)

    # =============================================================================
    # Mandatory methods
    # =============================================================================
    @staticmethod
    def generate(num_genes, generation_type):
        #Crear un switch de los diferentes gen type
        #num genes = numero de paquetes
        solution = []
        rango = range(num_genes)
        solution = random.sample(rango, num_genes)
        """while len(solution) != num_genes: # en vez d erandom range(num paquetes) -> shuffle // random.sample
            p= random.randint(0, num_genes-1)
            if p not in solution:
                solution.append(p)
        return solution"""
        return solution
    
        

    def __copy__(self):
        a = type(self)(self.num_genes, self.generation_type, self.crossover_type, self.mutation_type)
        a.solution = self.solution.copy()
        return a
    
    def crossover(self, individual):
        #Cruce 2PCX
        hijo1  = self.__copy__()
        hijo2 =  individual.__copy__()

        #Puntos de cruce random
        l = random.randint(0,self.num_genes-2) #Dos random distintos l<r # El -1 es 0
        r = random.randint(l+1, self.num_genes-1)
        #Copiamos parte cruzada
        index1 = []
        index2 = []
        for i in range(l+1,r+1):  #Con conjuntos, lo convertimos y miramos directamente ya que son tablas
            index1.append(individual.solution.index(self.solution[i]))
            index2.append(self.solution.index(individual.solution[i]))
        for i, idx in enumerate(sorted(index1)):
            hijo1.solution[i+l+1] = individual.solution[idx]
        for i, idx in enumerate(sorted(index2)):
            hijo2.solution[i+l+1] = self.solution[idx]
        return hijo1, hijo2
    
    def mutation(self):
        #Posiciones a mutar
        pos1 = random.randint(0, self.num_genes-1) #Dos random distintos
        pos2 = pos1
        while pos2 == pos1:
            pos2 = random.randint(0, self.num_genes-1)

        #Numero por el que cambias las posiciones
        n1 = self.solution[pos1]
        n2 = self.solution[pos2]
        #Hacemos el cambio
        self.solution[pos1] = n2
        self.solution[pos2] = n1
        
        return self

    def __lt__(self, item):
        return self.fitness < item.fitness

    def evaluate(self, evaluacion): #Llama al evaluate de la CVRP
        self.fitness = evaluacion.evaluate(self.solution)
        return self.fitness

    # =============================================================================
    # Recommended methods
    # =============================================================================
    
    def __repr__(self):
        return str([x for x in self.solution])

    pass



# **Se recomienda que se prueben cada uno de los métodos implementados de manera individual en las siguientes líneas de código:**


# TODO: Test here the methods to cross individuals

# TODO: Test here the methods to mutate an individual

# #### Clase `Genetic`
# 
# Esta clase implementa un **esquema** básico de **algoritmo genético**.
# 
# Los **métodos obligatorios** que se deben añadir son:
# 
# * ``def __init__(self, population_size, num_generations, selection_type, crossover_type, crossover_probability, mutation_type, mutation_probability, keep_elitism, random_state)``: Inicializa el **tamaño** de la **población**, el **tipo** de **operación** de **selección**, **cruce**, y **mutación**, así como la **probabilidad** de aplicar las operaciones de **cruce** y **mutación**. Además, también inicializa el **número** de **mejores soluciones** de la **población actual** que se **mantienen** en la **siguiente población** y una **semilla** para garantizar que los **experimentos** son **reproducibles**. **Nótese que puede ser necesario añadir más argumentos si así se requiere**.
# * ``def __call__(self, problem)``: **Método** que se **ejecuta** cuando se llama a un **objeto** de la **clase como** si fuese una **función**. En este **método** se debe **implementar** el **esquema básico** de un **algoritmo genético** que se encargue de ejecutar los pasos correspondientes. 
# * ``def generate_population(self, problem)``: **Genera** la **población inicial** de acuerdo con el **problema** a resolver.
# * ``def select_population(self, population, scores)``: **Selecciona** los **padres** a utilizar para la operación de cruce.
# * ``def crossover(self, population)``: **Cruza pares** de **padres** teniendo en cuenta la probabilidad de cruce.
# * ``def mutation(self, population)``: **Muta** los **individuos cruzados** teniendo en cuenta la probabilidad de mutación.
# * ``def evaluate(self, population, problem)``: **Evalua** los **nuevos individuos** de acuerdo con el problema a resolver.
# * ``def combine(self, population)``: **Forma** la **nueva generación** de acuerdo con el número de mejores individuos de la población actual a mantener en la siguiente.


import numpy as np


class Genetic:

    # =============================================================================
    # Constructor
    # =============================================================================
    def __init__(self, population_size, num_generations, selection_type, crossover_type, crossover_probability, mutation_type, mutation_probability, keep_elitism, random_state):
        self.population_size = population_size
        self.num_generations = num_generations #cuantas mas mejor
        self.selection_type = selection_type#no sirve
        self.crossover_type = crossover_type #no sirve
        self.crossover_probability = crossover_probability
        self.mutation_type = mutation_type #no sirve
        self.mutation_probability = mutation_probability
        self.keep_elitism = keep_elitism
        self.random_state = random_state
        random.seed(self.random_state)
        self.individuals_counter = 0


        

    # =============================================================================
    # Mandatory methods
    def generate_population(self, problem):
        population = []
        example = CVRP(problem,'a')
        num_genes = example.getGenes()
        for i in range(self.population_size):
            self.individuals_counter+=1
            individuo = Individual(num_genes, 'a', self.crossover_type,self.mutation_type)
            population.append(individuo)
        return population

    
    def select_population(self, population, scores):
        # Torneo
        # Participantes en el sorteo
        elegidos =  []
        if self.selection_type == "torneo":
            k = 5
            for i in range(self.population_size):
                candidatos = random.sample(range(self.population_size), k)
                min_score = np.inf
                for candidato in candidatos:
                    if scores[candidato] < min_score:
                        elegido = candidato
                        min_score = scores[candidato]
                elegidos.append(population[elegido])
        elif self.selection_type == "proporcional": #Haces la proporcion con la inversa y haces la probabilidad
            items = i = 0
            total = sum(scores)
            while items < self.population_size:
                n = random.uniform(0,1)
                if n < 1-scores[i]/total:
                    elegidos.append(population[i])
                    items += 1
                i += 1 
                i %= self.population_size
        return elegidos

        
    def crossover(self, population):
        result = []
        for i in range(0, self.population_size, 2):
            number = random.uniform(0,1)
            padre1 = population[i]
            padre2 = population[i+1]
            if number < self.crossover_probability:
                self.individuals_counter+=2
                hijo1, hijo2 = padre1.crossover(padre2)
                result.append(hijo1)
                result.append(hijo2)
            else:
                result.append(padre1)
                result.append(padre2)
        return result
    

    def mutation(self, population):
        for i in range(len(population)):
            number = random.uniform(0,1)
            if number < self.mutation_probability:
                self.individuals_counter+=1
                population[i] = population[i].mutation()
            
        return population

    def evaluate(self, population):
        score = [-1] * self.population_size
        for i in range(self.population_size):
            score[i] = population[i].evaluate(evaluacion=self.cvrp)
        return score

    def combine(self, new_population, new_score, population, score):
        mejores_antigua = [x for _, x in sorted(zip(score, population))][:self.keep_elitism]
        mejores_nueva = [x for _, x in sorted(zip(new_score, new_population))][:len(new_population)-self.keep_elitism]
        nueva_poblacion = mejores_nueva + mejores_antigua 
        return nueva_poblacion
    
    def __call__(self, problem):
        self.cvrp = CVRP(filename=problem, algorithm='-') #Esto se hace fuera 
        # 1.- t = 0
        # 2.- inicializar P(t)
        t1 = t2 = t3 = t4 = t0 = 0
        init_time = time.time()
        now = time.time()
        now = time.time()
        population = self.generate_population(problem) 
        t0 = time.time() -now
        # 3.- evaluar P(t)
        now = time.time()
        evaluation = self.evaluate(population)
        t1 = time.time() -now
        # 4.- Mientras (no se cumpla la condición de parada) hacer
        # 4.1.- t = t + 1
        for t in range(self.num_generations):
            # 4.2.- seleccionar P’(t) desde P(t-1)
            # Selección por torneo
            population_prime = self.select_population(population, evaluation)
            # Cruce 
            population_prime = self.crossover(population_prime)
            # 4.4.- mutación P’(t)
            # Mutación
            population_prime = self.mutation(population_prime)
            # 4.5.- evaluar P’(t)
            # Evaluamos cruce
            now = time.time()
            evaluation_prime = self.evaluate(population_prime)
            t2 +=time.time()-now
            # 4.6.- P(t) = combinar(P’(t), P(t-1))
            # Sustitución
            now = time.time()
            population = self.combine(population_prime,evaluation_prime, population, evaluation) 
            t3 += time.time() - now
            now = time.time()
            evaluation =  self.evaluate(population)
            t4 += time.time() - now
        print("Problema: ", problem)
        print("Tamaño población: ", self.population_size, "Numero de Generaciones: ", self.num_generations)
        print("Tipo de Selección: ", self.selection_type)
        print("Tipo de Cruce: ", self.crossover_type, "  Probabilidad: ", self.crossover_probability)
        print("Tipo de Mutación: ", self.mutation_type, " Probabilidad: ", self.mutation_probability)
        print("Elitismo: ", self.keep_elitism, " Seed: ", self.random_state)
        print("Solución: ", min(evaluation))
        print("Tiempo: ", time.time()-init_time)
        print("Nodos Reales: ", self.cvrp.get_nodos_reales(), "Nodos Totales: ", self.cvrp.get_nodos_totales())
        print("Individuos Reales: ", self.cvrp.get_real_individuals(), "Individuos Totales: ", self.individuals_counter)
        print("-----------------------------ESTADISTICAS---------------------------------------")
        print("Población inicial=",t0)
        print("1ª evaluación=",t1)
        print("Evaluaciones prime=",t2)
        print("Combinado=",t3)
        print("Evaluaciones poblacion=",t4)
        print("-------------------------------------------------------------------------------------------------")

gen = Genetic(population_size=50, num_generations=100, selection_type='proporcional', 
              crossover_type='2PCX', crossover_probability=0.9, 
              mutation_type='swap', mutation_probability=0.1, keep_elitism=5, random_state=0)
gen("/data/notebook_files/example.json")
gen("/data/notebook_files/small.json")
gen("/data/notebook_files/medium.json")
gen("/data/notebook_files/large.json")

"""
1ª evaluación= 0.959557056427002
Evaluaciones prime= 0.008309364318847656
Combinado= 0.011700630187988281
Evaluaciones poblacion= 0.0069887638092041016
-------------------------------------------------------------------------------------------------
Problema:  /data/notebook_files/small.json
Tamaño población:  50 Numero de Generaciones:  100
Tipo de Selección:  torneo
Tipo de Cruce:  2PCX   Probabilidad:  0.9
Tipo de Mutación:  swap  Probabilidad:  0.1
Elitismo:  5  Seed:  0
Solución:  12445.650000000001
Tiempo:  86.08556389808655
Nodos Reales:  189 Nodos Totales:  241200
Individuos Reales:  683 Individuos Totales:  10130
-----------------------------ESTADISTICAS---------------------------------------
Población inicial= 0.0075910091400146484
1ª evaluación= 83.28218245506287
Evaluaciones prime= 0.7348935604095459
Combinado= 0.013062477111816406
Evaluaciones poblacion= 0.022297143936157227
-------------------------------------------------------------------------------------------------
Problema:  /data/notebook_files/medium.json
Tamaño población:  50 Numero de Generaciones:  100
Tipo de Selección:  torneo
Tipo de Cruce:  2PCX   Probabilidad:  0.9
Tipo de Mutación:  swap  Probabilidad:  0.1
Elitismo:  5  Seed:  0
Solución:  36262.060000000005
Tiempo:  112.52370405197144
Nodos Reales:  255 Nodos Totales:  994950
Individuos Reales:  1085 Individuos Totales:  15191
-----------------------------ESTADISTICAS---------------------------------------
Población inicial= 0.1444714069366455
1ª evaluación= 97.56153297424316
Evaluaciones prime= 0.33197712898254395
Combinado= 0.04036116600036621
Evaluaciones poblacion= 0.030890941619873047
-------------------------------------------------------------------------------------------------
Problema:  /data/notebook_files/large.json
Tamaño población:  50 Numero de Generaciones:  100
Tipo de Selección:  torneo
Tipo de Cruce:  2PCX   Probabilidad:  0.9
Tipo de Mutación:  swap  Probabilidad:  0.1
Elitismo:  5  Seed:  0
Solución:  77201.68999999999
Tiempo:  2557.48544216156
Nodos Reales:  484 Nodos Totales:  442200
Individuos Reales:  1120 Individuos Totales:  20281
-----------------------------ESTADISTICAS---------------------------------------
Población inicial= 0.05882906913757324
1ª evaluación= 2274.271897792816
Evaluaciones prime= 279.9001762866974
Combinado= 0.013327836990356445
Evaluaciones poblacion= 0.02472996711730957


Problema:  /data/notebook_files/parcels1.json
Tamaño población:  50 Numero de Generaciones:  100
Tipo de Selección:  torneo
Tipo de Cruce:  2PCX   Probabilidad:  0.9
Tipo de Mutación:  swap  Probabilidad:  0.1
Elitismo:  5  Seed:  0
Solución:  34121.73
Tiempo:  6525.402714252472
Nodos Reales:  192 Nodos Totales:  341700
Individuos Reales:  975 Individuos Totales:  25358
-----------------------------ESTADISTICAS---------------------------------------
Población inicial= 0.05479884147644043
1ª evaluación= 6524.089193582535
Evaluaciones prime= 0.08461785316467285
Combinado= 0.011265039443969727
Evaluaciones poblacion= 0.010965108871459961
-------------------------------------------------------------------------------------------------
Problema:  /data/notebook_files/parcels2.json
Tamaño población:  50 Numero de Generaciones:  100
Tipo de Selección:  torneo
Tipo de Cruce:  2PCX   Probabilidad:  0.9
Tipo de Mutación:  swap  Probabilidad:  0.1
Elitismo:  5  Seed:  0
Solución:  68235.56999999996
Tiempo:  2688.152715921402
Nodos Reales:  48 Nodos Totales:  241200
Individuos Reales:  1359 Individuos Totales:  30478
-----------------------------ESTADISTICAS---------------------------------------
Población inicial= 0.031013965606689453
1ª evaluación= 2687.312640428543
Evaluaciones prime= 0.05927705764770508
Combinado= 0.012507915496826172
Evaluaciones poblacion= 0.009945154190063477
-------------------------------------------------------------------------------------------------
Problema:  /data/notebook_files/parcels3.json
Tamaño población:  50 Numero de Generaciones:  100
Tipo de Selección:  torneo
Tipo de Cruce:  2PCX   Probabilidad:  0.9
Tipo de Mutación:  swap  Probabilidad:  0.1
Elitismo:  5  Seed:  0
Solución:  55988.54
Tiempo:  4215.13051199913
Nodos Reales:  535 Nodos Totales:  492450
Individuos Reales:  942 Individuos Totales:  35552
-----------------------------ESTADISTICAS---------------------------------------
Población inicial= 0.02153944969177246
1ª evaluación= 3947.1186311244965
Evaluaciones prime= 264.9256148338318
Combinado= 0.012199878692626953
Evaluaciones poblacion= 0.025379657745361328
-------------------------------------------------------------------------------------------------

gen("/data/notebook_files/parcels1.json")
gen("/data/notebook_files/parcels2.json")
gen("/data/notebook_files/parcels3.json")
gen("/data/notebook_files/parcels4.json")
gen("/data/notebook_files/parcels5.json")
gen("/data/notebook_files/parcels6.json")

gen("/data/notebook_files/parcels7.json")
gen("/data/notebook_files/parcels8.json")
gen("/data/notebook_files/parcels9.json")
gen("/data/notebook_files/parcels10.json")

gen("/data/notebook_files/competition1.json")
gen("/data/notebook_files/competition1.json")
gen("/data/notebook_files/competition1.json")
gen("/data/notebook_files/competition1.json")

# **Se recomienda que se prueben cada uno de los métodos implementados de manera individual en las siguientes líneas de código:**


# #### Clase `CVRP`
# 
# Esta clase representa el problema en cuestión, esto es, el **problema** de **ruteo** de **vehículos**.
# 
# Los **métodos obligatorios** que se deben añadir son:
# 
# * ``def __init__(self, filename, algorithm)``: Inicializa el **problema** en cuestión y el **algoritmo** a usar para resolverlo. A su vez, se debe crear un **diccionario** que contenga como **clave** un **identificador** de **paquete** y como **valor** una **tupla** con la **ciudad** donde se encuentra dicho paquete y su **peso**.
# * ``def __call__(self)``: **Resuelve** el **problema** en cuestión.
# * ``def evaluate(self, solution)``: **Evalua** una **solución** para el **problema** en cuestión, **teniendo en cuenta** las **restricciones correspondientes**.
# * ``def search(self, departure, goal)``: **Resuelve** un **problema** de **búsqueda** de **caminos** dada las **ciudades** de **salida** y **meta**.
# 
# **Nótese que se puede crear una estructura de datos para agilizar el proceso de búsqueda de caminos requerido por el algoritmo ¿cuál puede ser?**


# TODO: Introduce here the answer to use for the hashing
answer = "cache"

# Avoid case sensitivity in the answer
answer = str.lower(answer)

# Encode the answer before hashing
answer = answer.encode("utf-8")

encoded = "0fea6a13c52b4d4725368f24b045ca84"  # The hashed answer
print(encoded)

check_answer(answer, encoded)

class CVRP:

    # =============================================================================
    # Constructor
    # =============================================================================
    def __init__(self, filename, algorithm): 
        self.filename = filename
        self.algorithm = algorithm
        
        #Method to read the problem JSON file
        with open(filename, 'r', encoding='utf8') as file:
            problem = json.load(file)

        #Crea diccionario de ciudades
        self.cities = {city['id']: city for city in problem['map']['cities']}

        #Crea diccionario de paquetes
        self.parcels = {parcel['id']: (parcel['city'],parcel['weight']) for parcel in problem['parcels']}
        self.warehouse = problem['warehouse']
        self.vehicles = {v['id']: v['capacity'] for v in problem['vehicles']}
        self.roads = problem['map']['roads']
        self.map = problem['map']
        #self.weight_car = problem['vehicles']['capacity']
        self.individuals_cache = {}
        self.routes_cache = {}
        self.counter_nodos = 0
        


    # =============================================================================
    # Mandatory methods
    # =============================================================================
    def __call__(self):
        #Nº de paquetes
        fitness = self.algorithm()
        print(fitness)
    
    def get_real_individuals(self):
        return len(self.individuals_cache)
    
    def get_nodos_reales(self):
        return len(self.routes_cache)
    
    def get_nodos_totales(self):
        return self.counter_nodos
    
    def get_km_ruta(self, origen, destino):
        if (origen, destino) in self.routes_cache or (destino, origen) in self.routes_cache:
            return self.routes_cache[(origen, destino)]
        km = self.search(origen, destino)  #Metemos las ciudades entre paquetes y las añadimos a los km recorridos
        self.routes_cache[(origen, destino)] = km
        self.routes_cache[(destino, origen)] = km
        return km

    def evaluate(self, solution): 
        self.counter_nodos +=len(solution)-1
        if tuple(solution) in self.individuals_cache:
            return self.individuals_cache[tuple(solution)]
        #Saco el primer destino
        # primero = self # paquete
        origen_city = self.warehouse # ciudad del paquete
        # primero_peso = self.parcels[primero][1] # peso del paquete ok 
        km_totales = 0
        #Hago el viaje al primer paquete
        # km_totales = self.get_km_ruta(self.warehouse,primero_city) 
        peso_total = 0 # primero_peso #Añado el peso del primer paquete

        for i in range(len(solution)):
            #paquete origen
            # origen = solution[i-1] # paquete
            # origen_city = self.parcels[origen][0] # ciudad del paquete
            #paquete destino
            destino = solution[i] # paquete
            destino_city = self.parcels[destino][0] # ciudad del paquete
            destino_peso = self.parcels[destino][1] # peso del paquete
            #Evaluamos el peso del vehículo actual 
            if peso_total > 20 : # Si es menor que 20 hacemos el viaje a por el siguiente paquete
                km = self.get_km_ruta(origen_city,self.warehouse)  + self.get_km_ruta(self.warehouse, destino_city)
                peso_total = 0
            else:
                km = self.get_km_ruta(origen_city, destino_city)
            km_totales += km
            peso_total += destino_peso  #Tambien añadimos el peso del siguiente paquete
            origen_city = destino_city
        #Saco el ultimo destino
        ultimo = solution[-1] # paquete
        ultimo_city = self.parcels[ultimo][0] # ciudad del paquete
        km_totales += self.get_km_ruta(ultimo_city, self.warehouse)
        self.individuals_cache[tuple(solution)] = km_totales 


        return km_totales


    def search(self, departure, goal): 
        
        data = {}
        data['goal'] = goal
        data['departure'] = departure
        data['map'] = self.map
        
        
        problem = Problem(data)
        var_her = HeuristicaGeodesica(problem)
        variable = AStar(problem,var_her)
        variable.do_search()
        km = variable.cost
        return km

        
    def getGenes(self):
        return len(self.parcels)

 
        
       

# **Se recomienda que se prueben cada uno de los métodos implementados de manera individual en las siguientes líneas de código:**


# TODO: Test here the method to initialize the capacited vehicle routing problem

# TODO: Test here the method to solve the search problem

# TODO: Test here the method to solve the capacited vehicle routing problem

# ### 4.3. Estudio y mejora de los algoritmos
# 
# Una vez que los algoritmos han sido implementados, se debe **estudiar** su **rendimiento**. Para ello, se debe comparar la **calidad** de las **soluciones obtenidas**, así como las **diferentes estadísticas** que se consideren adecuadas (número de generaciones, tiempo de ejecución, etc.). Factores como el tamaño máximo del problema que se soporta sin un error de memoria, así como el efecto temporal de usar escenarios más complejos son otros factores a tener en cuenta. Además, se **pueden proponer** y se valorarán la incorporación de **técnicas** que **permitan acelerar** la **ejecución** de los **algoritmos**.
# 
# ---


# TODO: Experiment here with the small problem

# TODO: Experiment here with the medium problem

# TODO: Experiment here with the large problem

# ### 5. Entrega y evaluación
# 
# Al igual que la práctica anterior, esta se debe **hacer en pares**. No obstante, en **casos excepcionales** se permite realizarla **individualmente**. **La fecha límite para subir la práctica es el 18 de diciembre de 2022 a las 23:55**. Las **entrevistas y evaluaciones** se realizarán la **semana siguiente**.
# 
# Algunas consideraciones:
# 
# * **En caso de que no se haya entregado la primera práctica, o se haya sacado menos de un cuatro, se podrán entregar conjuntamente en esta fecha. No obstante, se considerará únicamente un 90% de la nota global de prácticas**.
# * Está práctica supone el **70%** de la **nota** en este apartado.
# * La práctica se **evaluará** mediante una **entrevista individual** con el profesorado. Las fechas de las entrevistas se publicarán con antelación.
# *  Se proporcionará un **conjunto** de **casos** de **prueba preliminares** (varios mapas e instancias) que se **deben resolver correctamente**. En caso contrario, la práctica se considerará suspensa.
# * La **entrevista** consistirá en una serie de **preguntas** acerca del **código**.
# 
# **Por último, para la evaluación no continua se requirirá la implementación del algoritmo de búsqueda por ascenso de colinas. Además, este se deberá utilizar para inicializar la población del algoritmo genético, en lugar de que sea aleatoria.**


