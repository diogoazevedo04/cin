<h3 align="center">Universidade do Minho <br> Mestrado em Inteligência Artificial <br> Computação Inspirada na Natureza <br> 2024/2025 </h3>

---

<h3 align="center"> Colaboradores </h3>

<div align="center">

| Nome                    | Número |
|-------------------------|--------|
| Diogo José Borges Dias  | PG60245 |
| Diogo Lopes Azevedo     | PG61217 |

</div>

---

## Estrutura do Projeto

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
│   ├── Figure_1.png
|   ├── graph_spatial.png
|   └── hypervolume_over_generations.png
│
├── output/
│   ├── graph_base.gpickle
│   ├── moead_results.pkl
│   ├── pareto_front.csv
│   ├── evaluation_results.pkl
│   ├── evaluation_results.csv
│   ├── scenarios.pkl
│   ├── scenarios.csv
│   └── graph.html
│
├── src/                 
│   ├── graph.py          # Construção do grafo multimodal
│   ├── moead.py
│   ├── export_graph_html.py
│   ├── main.py
│   ├── scenarios.py
│   ├── evaluate_scenarios.py
│   ├── interactive_pareto.py
│   └── visualize.py
│
└── README.md

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

## Exportação do Grafo para Visualização HTML

O ficheiro `export_graph_html.py` gera uma visualização interativa do grafo multimodal em formato HTML.

**Características principais:**

* Visualização espacial da rede de transportes (Metro + STCP + Rede pedonal)
* Representação de nós (estações/paragens) e arestas (ligações)
* Interatividade: zoom, pan, seleção de nós
* Integração com informações de cada parada

**Execução:**

```bash
python src/export_graph_html.py

```

**Saída:**
* `output/graph.html` - Visualização interativa do grafo

---

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

## Geração de Cenários de Teste

O ficheiro `scenarios.py` permite gerar cenários de teste representativos para avaliação sistemática do algoritmo.

**Características principais:**

* Geração automática de 12 cenários com 4 níveis de dificuldade
* Dificuldade baseada em distância geográfica haversine
* Categorias: fácil (0.5-2.5 km), médio (2.5-5.0 km), difícil (5.0-10.0 km), muito difícil (10.0-20.0 km)

**Execução:**

```bash
python src/scenarios.py

```

**Saída:**
* `output/images/scenarios.pkl` - Cenários em formato binário
* `output/images/scenarios.csv` - Cenários em formato tabular (coordenadas, distâncias)

---

## Avaliação dos cenários

O ficheiro `evaluate_scenarios.py` executa uma avaliação do MOEA/D em todos os 12 cenários.

**Características principais:**

* Testa o algoritmo em condições variadas (diferentes distâncias e complexidade)
* Coleta métricas agregadas: tamanho da frente de Pareto, tempos min/max/avg, emissões min/max/avg
* Resumo por nível de dificuldade

**Execução:**

```bash
python src/evaluate_scenarios.py

```

**Saída:**
* `output/images/evaluation_results.pkl` - Resultados completos em formato binário
* `output/images/evaluation_results.csv` - Estatísticas por cenário (tabulado)

---

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

## Visualização Interativa de Pareto

O ficheiro `interactive_pareto.py` permite explorar a frente de Pareto gerada de forma interativa.

**Características principais:**

* Visualização web da frente de Pareto
* Navegação interativa entre soluções
* Exibição de detalhes de cada rota (tempo, CO₂, modo de transporte)

**Execução:**

```bash
python src/interactive_pareto.py

```

**Saída:**
* Interface web para análise interativa dos resultados

---