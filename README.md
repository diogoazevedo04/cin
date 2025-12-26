# Estrutura do Projeto

```text
.
├── data/
│   ├── gtfs/
│   │   ├── mdp/              # Dados GTFS do Metro do Porto
│   │   └── stcp/             # Dados GTFS da STCP
│   ├── output/
│   │   ├── graph_base.gpickle
│   │   ├── moead_results.pkl
│   │   └── pareto_front.csv
│
├── figures/                  # Figuras geradas para o relatório
│   ├── pareto_front.png
│   ├── tradeoff_extremes.png
│   ├── pareto_convergence.png
│   ├── time_distribution.png
│   ├── Figure_2.png
│   └── Figure_1.png
│
├── src                 # Construção do grafo multimodal
│   ├── graph.py
│   ├── moead.py
│   ├── lines.py
│   ├── main.py
│   └── visualize.py

```

## Dependências

O projeto foi desenvolvido em Python 3.9 ou superior. As principais dependências são:

* networkx
* numpy
* pandas
* matplotlib
* requests

**Instalação das dependências:**

```bash
pip install networkx numpy pandas matplotlib requests

```

## Construção do Grafo Multimodal

O ficheiro `graph.py` é responsável pela construção do grafo multimodal dirigido, integrando:

* Paragens de Metro
* Paragens de Autocarro
* Ligações pedonais entre modos diferentes

As arestas do grafo incluem informação detalhada sobre:

* Tempo de viagem
* Distância percorrida
* Emissões de CO₂
* Modo de transporte


**Para gerar o grafo base:**

```bash
python src/graph.py

```

O grafo é guardado em `data/output/graph_base.gpickle`.

## Otimização Multiobjetivo com MOEA/D

O ficheiro `moead.py` contém a implementação do algoritmo MOEA/D, adaptado ao problema de planeamento de percursos em grafos.

**Características principais:**

* Decomposição do problema multiobjetivo em subproblemas escalares
* Função de agregação de Tchebycheff
* Vizinhança definida no espaço dos vetores de peso
* Operadores genéticos específicos para grafos (crossover e mutação)

Cada solução representa um caminho válido no grafo, avaliado segundo:

1. Tempo total de viagem
2. Emissões totais de CO₂

O algoritmo produz uma aproximação da Frente de Pareto.

## Execução do Sistema

O ficheiro `main.py` funciona como ponto de entrada do sistema e orquestra todas as etapas:

1. Carregamento do grafo multimodal
2. Criação de nós virtuais de origem e destino
3. Execução do algoritmo MOEA/D
4. Análise das soluções obtidas
5. Exportação dos resultados

**Execução:**

```bash
python src/main.py

```

**Resultados gerados:**

* Frente de Pareto em formato CSV
* Resultados completos em formato pickle
* Estatísticas e análises no terminal

## Visualização dos Resultados

O ficheiro `visualize.py` permite gerar os gráficos utilizados na análise experimental e no relatório, incluindo:

* Frente de Pareto aproximada
* Compromisso entre tempo de viagem e emissões de CO₂
* Evolução da Frente de Pareto ao longo das gerações
* Distribuição dos tempos de viagem
* Visualização espacial da rede de transportes

**Execução:**

```bash
python src/visualize.py

```

As figuras são guardadas na pasta `figures/`.

## Contexto Académico

Este projeto foi desenvolvido no âmbito da unidade curricular **Computação Inspirada na Natureza**, do mestrado em Inteligência Artificial da Universidade do Minho.

Os principais conceitos abordados incluem:

* Otimização Multiobjetivo
* Dominância de Pareto
* Algoritmos Evolucionários
* MOEA/D
* Planeamento de percursos em grafos

## Autores

* Diogo José Borges Dias
* Diogo Lopes Azevedo