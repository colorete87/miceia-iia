# %%
import random

# %% [markdown]
"""
# Ejercicio Módulo 3 - N reinas
**Inteligencia Artificial - CEIA - FIUBA**

**INSERTE AQUÍ SU NOMBRE**

El problema de las **N reinas** es un clásico desafío en informática que consiste en colocar N reinas sobre un tablero de ajedrez de tamaño N x N de manera que ninguna reina pueda atacar a otra.

### ¿Qué significa que las reinas no se ataquen?

En ajedrez, una reina puede moverse y atacar en cualquier cantidad de casillas:

- En la misma fila
- En la misma columna
- En cualquiera de las diagonales

Por lo tanto, la restricción principal del problema es que no puede haber dos reinas compartiendo:

- La misma fila
- La misma columna
- La misma diagonal

### ¿Qué vamos a hacer en este ejercicio?

En este ejercicio trabajaremos con el clásico problema de las N reinas y exploraremos cómo encontrar una solución usando métodos de búsqueda local. Primero, veremos una implementación básica basada en **gradiente descendente discreto (hill climbing)**, que nos permitirá entender cómo movernos en el espacio de soluciones para minimizar conflictos.

Luego, tu desafío será implementar el algoritmo de **Simulated Annealing**, una técnica que mejora la búsqueda local al evitar quedar atrapados en soluciones subóptimas.


Para lograr esto, iremos paso a paso, definiendo y trabajando con:
1. **Representación del problema en Python**: cómo codificamos el tablero y las posiciones de las reinas.
2. **Generación de vecinos**: cómo obtener estados cercanos a partir de un estado actual.
3. **Función de costo**: cómo medir la calidad de una solución (número de conflictos).
4. **Visualización del tablero**: mostrar las reinas ubicadas para entender mejor cada estado.
5. **Algoritmo de gradiente descendente**: aplicar esta búsqueda local para acercarnos a una solución.


## Representación del problema en Python

Para trabajar con el problema de las N reinas, necesitamos una forma sencilla y eficiente de representar el tablero y la ubicación de las reinas.

La forma más común y práctica es usar una lista de tamaño N, donde:
- El índice de la lista representa la columna del tablero.
- El valor almacenado en ese índice representa la fila donde está ubicada la reina en esa columna.

Por ejemplo, para *N=8*, un arreglo como este:
"""

# %%
board = [0, 4, 7, 5, 2, 6, 1, 3]

# %% [markdown]
"""
Significa:
- La reina en la columna 0 está en la fila 0.
- La reina en la columna 1 está en la fila 4.
- La reina en la columna 2 está en la fila 7.
- ... y así sucesivamente.

De esta forma podemos representar diferentes estados y empezar a "mover" a las reinas. Pero antes, necesitamos una función que inicialice un estado al azar:
"""

# %%
def initialize_random_state(n: int = 8):
    """
    Initializes a random state for the N-queens problem.
    Returns a list where each index represents a column
    and the value is the row where the queen is placed.
    """
    return [random.randint(0, n - 1) for _ in range(n)]

# Ejemplo de uso:
estado_inicial = initialize_random_state()
print("Estado inicial aleatorio:", estado_inicial)

# %% [markdown]
"""
## Generación de vecinos

En los algoritmos de búsqueda local, la idea es moverse en el espacio de soluciones explorando vecinos del estado actual.

**¿Qué es un vecino en el problema de las N reinas?**

Dado un estado (una configuración particular de reinas en el tablero), un vecino es otro estado que resulta de hacer una pequeña modificación en el estado actual.

En nuestro caso, donde cada reina está en una columna, una forma natural de definir vecinos es:

1. Elegir una columna.
2. Mover la reina de esa columna a una fila diferente dentro de la misma columna.

**Ejemplo**

Si el estado actual es:
"""

# %%
board = [0, 4, 7, 5, 2, 6, 1, 3]

# %% [markdown]
"""
Un vecino puede ser:

`Mover la reina en la columna 2 (que está en fila 7) a la fila 0`, dando:
"""

# %%
neig_board = [0, 4, 0, 5, 2, 6, 1, 3]

# %% [markdown]
"""
A continuación, definamos una función que genera todos los vecinos posibles a partir de un estado dado:
"""

# %%
def generate_neighbors(state):
    """
    Generates all neighboring states from the current state.
    A neighbor is a state where one queen is moved to a different
    row within its column.
    
    Args:
        state (list): List representing the queen positions.
        
    Returns:
        list: List of neighboring states.
    """
    n = len(state)
    neighbors = []
    
    for col in range(n):
        current_row = state[col]
        for new_row in range(n):
            if new_row != current_row:
                neighbor = state.copy()
                neighbor[col] = new_row
                neighbors.append(neighbor)
                
    return neighbors

# Ejemplo de uso:
estado_actual = [0, 4, 7, 5, 2, 6, 1, 3]
vecinos = generate_neighbors(estado_actual)
print(f"Cantidad de vecinos generados: {len(vecinos)}")
print("Algunos vecinos:")
for vecino in vecinos[:5]:
    print(vecino)

# %% [markdown]
"""
**¿Qué hace esta función?**

1. Para cada columna, prueba mover la reina a todas las filas distintas de la actual.
2. Copia el estado actual y modifica la posición para crear un vecino nuevo.
3. Devuelve una lista con todos los vecinos posibles.

## Función de costo

En los algoritmos de búsqueda local, cada estado (o solución parcial) tiene un *costo* que indica qué tan lejos está de ser una solución válida.

En el caso del problema de las N reinas, una solución válida es aquella donde ninguna reina puede atacar a otra. Por lo tanto, una forma natural de definir la función de costo es:

> Sea:
> - N el tamaño del tablero (y cantidad de reinas)
> - $s = [s_0, s_1, ..., s_{N-1}]$ un estado, donde $s_i \in \{ 0, 1, ..., N-1 \}$ representa la fila en la quie está la reina de la columna *i*.
>
> Entonces, la función de costo $C(s)$ mide el número total de pares de reinas que se atacan entre sí:
>
> $$C(s) = \sum_{i=0}^{N-1} \sum_{j=i+1}^{N-1} \delta(s_i, s_j)$$
> donde la función $\delta(s_i, s_j)$ se define como:
> $$ \delta(s_i, s_j) = \begin{cases} 1 & \text{si } s_i = s_j \quad \text{(misma fila)} \\ 1 & \text{si } |s_i - s_j| = |i - j| \quad \text{(misma diagonal)} \\ 0 & \text{en otro caso}\end{cases}  $$

Queremos minimizar esta función:
- Si el costo es `0`, ¡tenemos una solución válida!
- Si el costo es mayor que `0`, hay conflictos que resolver.

¿Cuándo dos reinas se atacan? Dos reinas se atacan si están:

- En la misma fila
- En la misma diagonal (ascendente o descendente)
- ⚠️ No necesitamos preocuparnos por columnas, porque la representación del problema (una reina por columna) ya garantiza que no haya más de una reina por columna.

La implementación en Python es:
"""

# %%
def cost(state):
    """
    Calculates the number of pairs of queens that are attacking each other.
    
    Args:
        state (list): A list where state[i] is the row of the queen in column i.
        
    Returns:
        int: Total number of conflicting pairs.
    """
    n = len(state)
    conflicts = 0
    for i in range(n):
        for j in range(i + 1, n):
            same_row = state[i] == state[j]
            same_diag = abs(state[i] - state[j]) == abs(i - j)
            if same_row or same_diag:
                conflicts += 1
    return conflicts

# Ejemplo de uso
state = [0, 4, 7, 5, 2, 6, 1, 3]  # Solución valida para N=8
print("Estado:", state) 
print("Costo:", cost(state))     

state = [0, 4, 7, 5, 2, 6, 1, 0]
print("Estado:", state)
print("Costo:", cost(state))

# %% [markdown]
"""
Esta función se usa para:
- Evaluar qué tan buena es una solución actual.
- Comparar varios vecinos y elegir el de menor costo.
- Saber si ya llegamos a una solución (cuando el costo es 0).

## Visualización del tablero

Una forma muy útil de comprender y depurar soluciones al problema de las N reinas es poder visualizar el estado del tablero. Para esto, podemos imprimirlo en consola usando caracteres ASCII, representando las casillas vacías y las reinas.

¿Cómo lo representamos? Usamos los siguientes símbolos:

- `♛` para indicar una reina
- `|` y `---` para simular los bordes del tablero

Definamos la función que nos permita armar la representación:
"""

# %%
def print_board_ascii(state: list):
    """
    Prints an ASCII representation of the N-Queens board.
    
    Each index in the input list represents the column,
    and the value at that index represents the row where the queen is placed.

    Args:
        state (list): A list of integers where state[i] is the row of the queen in column i.

    Example:
        For board = [0, 4, 7, 5, 2, 6, 1, 3], prints a visual 8x8 board with queens.
    """
    n = len(state)
    horizontal_border = "+" + "---+" * n
    for row in range(n):
        print(horizontal_border)
        row_str = ""
        for col in range(n):
            if state[col] == row:
                row_str += "| ♛ "
            else:
                row_str += "|   "
        row_str += "|"
        print(row_str)
    print(horizontal_border)

# Ejemplo de uso
state = [0, 4, 7, 5, 2, 6, 1, 3]  # Solución válida para N=8
print_board_ascii(state)

# %% [markdown]
"""
## Algoritmo de gradiente descendente (hill climbing)

Tal como vimos en los videos, este algoritmo parte de un estado inicial y se mueve siempre al mejor vecino, es decir, al estado vecino con menor cantidad de conflictos (menor costo), esperando llegar a una solución sin conflictos.

**Estructura general del algoritmo**
1. Generar un estado inicial aleatorio.
2. Evaluar su costo.
3. Generar todos sus vecinos.
4. Elegir el vecino con menor costo.
5. Si es mejor que el actual, avanzar a ese estado.
6. Si no hay mejora posible → nos quedamos en un óptimo local.

Veamos la implementación:
"""

# %%
N = 8 # Número de reinas
MAX_STEPS = 1000 # Número maximo de iteraciones

def grad_desc(n, max_steps=1000) -> tuple:
    """
    Hill Climbing algorithm for solving the N-Queens problem.
    
    Args:
        n (int): Size of the board (number of queens).
        max_steps (int): Maximum number of steps before giving up.
        
    Returns:
        tuple: (final_state, final_cost)
    """

    # 1. Inicializamos un estado al azar
    current = initialize_random_state(n)
    # Calculamos el costo
    current_cost = cost(current)

    print("Estado inicial:", current)
    print("Costo:", current_cost)
    print_board_ascii(current)
    print("------------")
    
    last_step = max_steps
    for step in range(max_steps):
        # 3. Generamos todos sus vecinos.
        neighbors = generate_neighbors(current)
        # Calculamos todos los costos de los vecinos
        neighbor_costs = [cost(neighbor) for neighbor in neighbors]
        # Obtenemos el minimo costo
        min_cost = min(neighbor_costs)

        # Si todos los vecinos tienen el mismo valor o mayor, terminamos.
        if min_cost >= current_cost:
            last_step = step
            break

        # Elegimos el vecino con menor costo.
        best_index = neighbor_costs.index(min_cost)
        current = neighbors[best_index]
        current_cost = min_cost
        

    if current_cost == 0:
        print("✅ ¡Se encontró una solución!")
    else:
        print("❌ No se encontró una solución :(")

    print("Iteraciones ejecutadas:", last_step)
    print("Estado final:", current)
    print("Costo final:", current_cost)
    print_board_ascii(current)

    return current, current_cost

# %%
_ = grad_desc(8, max_steps=MAX_STEPS)

# %%
_ = grad_desc(8, max_steps=MAX_STEPS)

# %%
_ = grad_desc(8, max_steps=MAX_STEPS)

# %%
_ = grad_desc(8, max_steps=MAX_STEPS)

# %%
_ = grad_desc(8, max_steps=MAX_STEPS)

# %% [markdown]
"""
De 5 ejecuciones, 3 termino en solución.

### Simmulated Annealing

Llegamos a la parte que **vos debés desarrollar**. Ahora te toca implementar el algoritmo de búsqueda Simulated Annealing.

Para eso, debés completar la función que aparece más abajo.

> ⚠️ No modifiques los argumentos ni el tipo de salida: la función debe recibir exactamente los parámetros indicados y devolver el resultado especificado.
"""

# %%
def simulated_annealing(n: int, initial_temp: float, cooling_rate: float, max_steps: int = 1000) -> tuple:
    """
    Simulated Annealing algorithm for solving the N-Queens problem.
    
    Args:
        n (int): Size of the board (number of queens).
        initial_temp (float): Starting temperature.
        cooling_rate (float): How much the temperature decreases per step.
        max_steps (int): Maximum number of iterations.
    
    Returns:
        tuple: (final_state, final_cost)
    """
    current = initialize_random_state(n)
    current_cost = cost(current)
    temperature = initial_temp

    print("Estado inicial:", current)
    print("Costo:", current_cost)
    print_board_ascii(current)
    print("------------")

    #######################
    # TODO: Implementa aquí el algoritmo de Simulated Annealing.
    import math

    if current_cost == 0:
        last_step = 0
    else:
        last_step = max_steps
        for step in range(max_steps):
            neighbors = generate_neighbors(current)
            amount_neib = len(neighbors)

            accepted = False
            while amount_neib > 0:
                neighbor = random.choice(neighbors)
                neighbors.remove(neighbor)
                amount_neib -= 1

                neighbor_cost = cost(neighbor)
                delta_cost = neighbor_cost - current_cost

                if delta_cost < 0:
                    current = neighbor
                    current_cost = neighbor_cost
                    accepted = True
                    break
                else:
                    if temperature > 0:
                        if random.random() < math.exp(-delta_cost / temperature):
                            current = neighbor
                            current_cost = neighbor_cost
                            accepted = True
                            break

            if not accepted:
                last_step = step
                break

            if current_cost == 0:
                last_step = step + 1
                break

            temperature *= cooling_rate
    #######################

    if current_cost == 0:
        print("✅ ¡Se encontró una solución!")
    else:
        print("❌ No se encontró una solución :(")

    print("Iteraciones ejecutadas:", last_step)
    print("Estado final:", current)
    print("Costo final:", current_cost)
    print_board_ascii(current)

    return current, current_cost 

# %% [markdown]
"""
Ejecutemos 5 veces la función:
"""

# %%
initial_temp = 1.0
cooling_rate = 0.95
_ = simulated_annealing(8, initial_temp, cooling_rate, max_steps=MAX_STEPS) # Cambia a initial_temp y cooling_rate con los valores que deseas usar.

# %%
initial_temp = 0.5
cooling_rate = 0.95
_ = simulated_annealing(8, initial_temp, cooling_rate, max_steps=MAX_STEPS) # Cambia a initial_temp y cooling_rate con los valores que deseas usar.

# %%
initial_temp = 0.25
cooling_rate = 0.95
_ = simulated_annealing(8, initial_temp, cooling_rate, max_steps=MAX_STEPS) # Cambia a initial_temp y cooling_rate con los valores que deseas usar.

# %%
initial_temp = 1.0
cooling_rate = 0.9
_ = simulated_annealing(8, initial_temp, cooling_rate, max_steps=MAX_STEPS) # Cambia a initial_temp y cooling_rate con los valores que deseas usar.

# %%
initial_temp = 1.0
cooling_rate = 0.8
_ = simulated_annealing(8, initial_temp, cooling_rate, max_steps=MAX_STEPS) # Cambia a initial_temp y cooling_rate con los valores que deseas usar.

# %% [markdown]
"""
Escribe aqui cuantas veces de las 5 ejecuciones se encontró la solucion: 5 de 5
"""

