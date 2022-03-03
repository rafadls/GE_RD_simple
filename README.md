# GE_RD_simple

**Install**

```bash
git clone https://github.com/rafadls/GE_RD_simple
cd GE_RD_simple
```

**Requirements**

```bash
pip install -r requirements.txt
```

**Variable**


To set the grid edit the "parameters/listaDeParametros.json" file
```
{
"variableParams": 
    {
    "COEFICIENTE": [1,2,3], 
    "POPULATION_SIZE": [50],
    "GENERATIONS": [50],
    "CROSSOVER_PROBABILITY": [0.75],
    "MUTATION_PROBABILITY": [0.15],
    "optimizeConstant_each":[50],
    "MR": [false, true],
    "Correlation": [false],
    "check_minimum_fitness": [false]
    }
}
```

**Run**

```bash
cd src/
python ponyge.py --variable
```

