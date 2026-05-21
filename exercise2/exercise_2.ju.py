# %% [markdown]
"""
# Ejercicio Módulo 2
**Inteligencia Artificial - CEIA - FIUBA**

**INSERTE AQUÍ SU NOMBRE**

En este ejercicio deben implementar un algoritmo de búsqueda que no sea **Búsqueda Primero en Anchura (BFS)** para resolver el problema de la Torre de Hanoi. La nota máxima dependerá del algoritmo implementado:

- **Búsqueda Primero en Profundidad**: nota máxima 6.
- **Búsqueda de Costo Uniforme**: nota máxima 6.
- **Búsqueda de Profundidad Limitada con Profundidad Iterativa**: nota máxima 7.
- **Búsqueda Voraz usando la heurística dada en el aula virtual**: nota máxima 8.
- **Búsqueda Voraz usando una heurística desarrollada por vos**: nota máxima 9.
- **Búsqueda A\* usando la heurística dada en el aula virtual**: nota máxima 9.
- **Búsqueda A\* usando una heurística desarrollada por vos**: nota máxima 10.

La función debe devolver la salida correspondiente a la solución encontrada o `None si no se encontró una solución.

Además, debe calcular métricas de rendimiento que, como mínimo, incluyan:

- `solution_found`: `True` si se encontró la solución, `False` en caso contrario.
- `nodes_explored`: cantidad de nodos explorados (entero).
- `states_visited`: cantidad de estados distintos visitados (entero).
- `nodes_in_frontier`: cantidad de nodos que quedaron en la frontera al finalizar la ejecución (entero).
- `max_depth`: máxima profundidad explorada (entero).
- `cost_total`: costo total para encontrar la solución (float).
"""

# %%
from aima_libs.hanoi_states import ProblemHanoi, StatesHanoi
from aima_libs.tree_hanoi import NodeHanoi
from aima_libs.aima import PriorityQueue as AimaPriorityQueue
import numpy as np


def search_algorithm_breadth(number_disks=5) -> (NodeHanoi, dict):
    """
    Anchura
    """

    # Inicializamos el problema
    list_disks = [i for i in range(number_disks, 0, -1)]
    initial_state = StatesHanoi(list_disks, [], [], max_disks=number_disks)
    goal_state = StatesHanoi([], [], list_disks, max_disks=number_disks)
    problem = ProblemHanoi(initial=initial_state, goal=goal_state)

    # Inicialización
    frontier = [NodeHanoi(problem.initial)]
    explored = set()
    node_explored = 0

    # Búsqueda
    while len(frontier) != 0:
        # Exploramos el primer nodo de la frontera
        node = frontier.pop()
        node_explored += 1
        explored.add(node.state)  # Verificamos si llegamos al objetivo

        # Encontramos el problema
        if problem.goal_test(node.state):
            metrics = {
                "solution_found": True,
                "nodes_explored": node_explored,
                "states_visited": len(explored),
                "nodes_in_frontier": len(frontier),
                "max_depth": node.depth,
                "cost_total": node.state.accumulated_cost,
            }
            return node, metrics

        # Agregamos a la frontera los nodos sucesores que no hayan sido visitados
        for next_node in node.expand(problem):
            # "Cache" de nodos
            if next_node.state not in explored:
                frontier.insert(0, next_node)

    # Si no se encuentra solución, devolvemos métricas igualmente
    metrics = {
        "solution_found": False,
        "nodes_explored": node_explored,
        "states_visited": len(explored),
        "nodes_in_frontier": len(frontier),
        "max_depth": node.depth,
        "cost_total": None,
    }

    return None, metrics


def search_algorithm_depth(number_disks=5) -> (NodeHanoi, dict):
    """
    Profundidad
    """

    # Inicializamos el problema
    list_disks = [i for i in range(number_disks, 0, -1)]
    initial_state = StatesHanoi(list_disks, [], [], max_disks=number_disks)
    goal_state = StatesHanoi([], [], list_disks, max_disks=number_disks)
    problem = ProblemHanoi(initial=initial_state, goal=goal_state)

    # Inicialización
    frontier = [NodeHanoi(problem.initial)]
    explored = set()
    node_explored = 0

    # Búsqueda
    while len(frontier) != 0:
        # Exploramos el primer nodo de la frontera
        node = frontier.pop()
        node_explored += 1
        explored.add(node.state)  # Verificamos si llegamos al objetivo

        # Encontramos el problema
        if problem.goal_test(node.state):
            metrics = {
                "solution_found": True,
                "nodes_explored": node_explored,
                "states_visited": len(explored),
                "nodes_in_frontier": len(frontier),
                "max_depth": node.depth,
                "cost_total": node.state.accumulated_cost,
            }
            return node, metrics

        # Agregamos a la frontera los nodos sucesores que no hayan sido visitados
        for next_node in node.expand(problem):
            # "Cache" de nodos
            if next_node.state not in explored:
                frontier.append(next_node)

    # Si no se encuentra solución, devolvemos métricas igualmente
    metrics = {
        "solution_found": False,
        "nodes_explored": node_explored,
        "states_visited": len(explored),
        "nodes_in_frontier": len(frontier),
        "max_depth": node.depth,
        "cost_total": None,
    }

    return None, metrics


def search_algorithm(number_disks=5):
    """
    Heurística
    """

    def h_function(node, number_disks):
        """
        Heurística propia:
        - Creo un vector de tamaño number_disks x pegs: 0 si no hay disco, o
          número de disco en la posición
        - diferencia absoluta entre los vectores
          "estado actual" y "estado obejtivo"
        """

        # función para rellenar con ceros
        def pad_peg(peg):
            return list(peg) + [0] * (number_disks - len(peg))

        # estado actual
        actual_array = np.array(
            pad_peg(node.state.rods[0])
            + pad_peg(node.state.rods[1])
            + pad_peg(node.state.rods[2])
        )

        # estado objetivo
        goal_peg3 = [i for i in range(number_disks, 0, -1)]
        goal_array = np.array(pad_peg([]) + pad_peg([]) + pad_peg(goal_peg3))

        # métrica
        diferencia_total = np.sum(np.abs(actual_array - goal_array))

        return int(diferencia_total)

    def g_function(node):
        return node.state.accumulated_cost

    def cost_function(node):
        return h_function(node, number_disks) + g_function(node)

    # Inicializamos el problema
    list_disks = [i for i in range(number_disks, 0, -1)]
    initial_state = StatesHanoi(list_disks, [], [], max_disks=number_disks)
    goal_state = StatesHanoi([], [], list_disks, max_disks=number_disks)
    problem = ProblemHanoi(initial=initial_state, goal=goal_state)

    # Inicialización
    frontier = AimaPriorityQueue(order="min", f=cost_function)
    initial_node = NodeHanoi(problem.initial)
    frontier.append(initial_node)  # La cola automáticamente evalúa f(initial_node)
    explored = set()
    node_explored = 0

    # Búsqueda
    while len(frontier) != 0:
        # Extrae el nodo con el menor costo
        node = frontier.pop()[1]
        node_explored += 1
        explored.add(node.state)

        # Verificamos si llegamos al objetivo
        if problem.goal_test(node.state):
            metrics = {
                "solution_found": True,
                "nodes_explored": node_explored,
                "states_visited": len(explored),
                "nodes_in_frontier": len(frontier),
                "max_depth": node.depth,
                "cost_total": node.state.accumulated_cost,
            }
            return node, metrics

        # Agregamos a la frontera los nodos sucesores que no hayan sido visitados
        for next_node in node.expand(problem):
            # Verificamos si ya fue explorado o si ya está en la frontera con un costo menor
            if next_node.state not in explored and next_node not in frontier:
                frontier.append(next_node)

    # Si no se encuentra solución
    metrics = {
        "solution_found": False,
        "nodes_explored": node_explored,
        "states_visited": len(explored),
        "nodes_in_frontier": len(frontier),
        "max_depth": node.depth if "node" in locals() else 0,
        "cost_total": None,
    }

    return None, metrics


# %% [markdown]
"""
Veamos las métricas:
"""

# %%

n = 5

if n <= 5:
    print("Solución en anchura")
    solution, metrics = search_algorithm_breadth(n)
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print()

print("Solución en produnidad")
solution, metrics = search_algorithm_depth(n)
for key, value in metrics.items():
    print(f"{key}: {value}")
print()

solution, metrics = search_algorithm(n)
for key, value in metrics.items():
    print(f"{key}: {value}")
print()


# %%
# %%timeit
if n <= 5:
    solution, metrics = search_algorithm_breadth(number_disks=n)

# %%
# %%timeit
solution, metrics = search_algorithm_depth(number_disks=n)

# %%
# %%timeit
solution, metrics = search_algorithm(number_disks=n)

# %%
if n <= 5:
    import tracemalloc

    tracemalloc.start()

    solution, metrics = search_algorithm_breadth(number_disks=n)

    # Para medir memoria consumida usamos el pico de memoria
    _, memory_peak = tracemalloc.get_traced_memory()
    memory_peak /= 1024 * 1024
    tracemalloc.stop()

    print(
        f"Pico de memoria ocupada: {round(memory_peak, 2)} [MB]",
    )

# %%
import tracemalloc

tracemalloc.start()

solution, metrics = search_algorithm_depth(number_disks=n)

# Para medir memoria consumida usamos el pico de memoria
_, memory_peak = tracemalloc.get_traced_memory()
memory_peak /= 1024 * 1024
tracemalloc.stop()

print(
    f"Pico de memoria ocupada: {round(memory_peak, 2)} [MB]",
)

# %%
import tracemalloc

tracemalloc.start()

solution, metrics = search_algorithm(number_disks=n)

# Para medir memoria consumida usamos el pico de memoria
_, memory_peak = tracemalloc.get_traced_memory()
memory_peak /= 1024 * 1024
tracemalloc.stop()

print(
    f"Pico de memoria ocupada: {round(memory_peak, 2)} [MB]",
)

# %% [markdown]
"""
Veamos las métricas:
"""

# %%
for key, value in metrics.items():
    print(f"{key}: {value}")

# %% [markdown]
"""
Veamos el camino de estados desde el principio a la solución:
"""

# %%
for nodos in solution.path():
    print(nodos)

# %% [markdown]
"""
Y las acciones que el agente debería aplicar para llegar al objetivo:
"""

# %%
for act in solution.solution():
    print(act)

# %%
