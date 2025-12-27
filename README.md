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

## Software e Justificação das Escolhas Tecnológicas

### Linguagem de Programação

**Python 3.9+**: Escolhido pela sua versatilidade na computação científica, extensa comunidade em IA/otimização e ecossistema robusto de bibliotecas. A sintaxe clara facilita prototipagem rápida de algoritmos complexos.

### Bibliotecas Fundamentais

#### NetworkX
**Versão**: Atual  
**Utilização**: Construção, manipulação e análise de grafos multimodais dirigidos.  
**Justificação**: NetworkX oferece uma API robusta para grafos com suporte a atributos nas arestas e nós (tempo, CO₂, modo), essencial para a representação multimodal do sistema de transportes. É a biblioteca *de facto* em Python para teoria de grafos (Harris et al., 2020).

#### NumPy
**Versão**: Atual  
**Utilização**: Operações vetorizadas, geração de vetores de peso para decomposição MOEA/D, cálculos de normas euclidianas.  
**Justificação**: NumPy é fundamental para computação numérica eficiente. A operação `np.linalg.norm()` para cálculo de proximidade entre vetores de peso é otimizada em C, garantindo desempenho em problemas de grande escala.

#### Pandas
**Versão**: Atual  
**Utilização**: Leitura/escrita de ficheiros CSV (cenários, resultados), análise exploratória de dados.  
**Justificação**: Pandas simplifica o trabalho com dados tabulares e integra-se bem com o ecossistema científico Python. Facilita reprodutibilidade através de formato CSV legível.

#### Matplotlib
**Versão**: Atual  
**Utilização**: Geração de gráficos de análise (Frente de Pareto, hipervolume, distribuição de tempos).  
**Justificação**: Matplotlib é a biblioteca padrão para visualização em Python. Oferece controlo fino sobre a aparência dos gráficos, essencial para publicações académicas.

#### Folium
**Versão**: Atual  
**Utilização**: Visualização interativa de rotas em mapas geográficos (solução_map.py).  
**Justificação**: Folium integra-se com OpenStreetMap e permite criar mapas interativos com anotações de modo (walk/metro/bus), facilitando interpretação geográfica das soluções.

#### Requests
**Versão**: Atual  
**Utilização**: Download de dados GTFS de fontes web públicas.  
**Justificação**: Requests é a biblioteca HTTP padrão em Python, simples e confiável para obtenção de dados de APIs públicas.

### Algoritmo MOEA/D

**Referência Principal**: Zhang & Li (2007) - "MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition"

**Escolha Justificada**:
- Decomposição do problema multiobjetivo em subproblemas escalares via vetores de peso, permitindo exploração diversa da Frente de Pareto
- Cooperação entre subproblemas vizinhos melhora qualidade das soluções
- Eficiente em problemas com 2-3 objetivos (nosso caso: tempo e CO₂)
- Implementação modular facilita adaptação ao domínio de grafos

### Função de Agregação: Tchebycheff

**Justificação**: A agregação Tchebycheff (max ponderado) é mais robusta que weighted sum para frentes de Pareto não convexas. Garante cobertura uniforme da Frente em problemas de transporte multimodal, onde trade-offs não são lineares (Zhang & Li, 2007).

### Métricas de Avaliação

**Hipervolume** (cálculo 2D conforme Zitzler & Thiele, 1999):
- Mede simultaneamente convergência e diversidade da Frente de Pareto
- Aplicável a qualquer Frente sem conhecimento a priori da solução ótima
- Escolhido porque é independente de preferências de decisor

**Epsilon-dominância** (pruning de soluções redundantes):
- Remove soluções dominadas (diferença < ε) para evitar inchação de população
- Mantém aproximação compacta da Frente
- Valores: `epsilon_time=0.3 min`, `epsilon_co2=1.0 g` (calibrados empiricamente)

### Operadores Genéticos

**Crossover por Nó Comum**:
- Combina dois caminhos pelo nó mais próximo comum
- Mantém conectividade e validade de grafos
- Taxa: 80% (exploração suficiente sem perda de diversidade)

**Mutação por Subcaminho (Dijkstra)**:
- Substitui segmento aleatório de caminho por shortest-path ponderado
- Suporta dois modos:
  - *Orientado* (inicialização): edge_score por pesos MOEA/D → rotas coerentes com objetivo
  - *Exploratório* (gerações): aleatoriedade via jitter ±5% → fuga de óptimos locais
- Taxa: 80% (agressivo para evitar convergência prematura)

**Justificação**: Operadores respeitam estrutura de grafo (garantem caminhos válidos) e exploram espaço de soluções de forma não-aleatória via custos relevantes ao problema.

### Penalidades por Restrições

- **Transbordos > 4**: Penalidade 500 por excesso → desencoraja rotas fragmentadas
- **Tempo pedonal > 90 min**: Penalidade 100 por excesso → evita rotas inconvenientes

**Justificação**: Penalidades grande (500) vs moderada (100) refletem preferência maior por limitar transbordos que tempo pedonal, coerente com literatura de qualidade de serviço de transportes (UITP, 2017).

### Dados GTFS

**Fonte**: Metro do Porto e STCP (feeds públicos)  
**Justificação**: GTFS é formato standardizado (Google, 2005) para dados de transporte público. Permite reprodutibilidade e integração com ferramentas de análise (QGIS, SUMO, etc).

### Escolhas de Parâmetros MOEA/D

| Parâmetro | Valor | Justificação |
|-----------|-------|------------|
| Population Size | 100 | Trade-off entre diversidade de Frente (ideal > 50) e tempo computacional |
| Generations | 50 | Estabilização de hipervolume ~geração 30-35 conforme análise experimental |
| Neighbors (T) | 20 | Vizinhança 20% da população balanceia exploração local/global |
| Mutation Rate | 0.8 | Alto para evitar convergência prematura em problema complexo |
| Crossover Rate | 0.8 | Alto para manter diversidade de material genético |

### Ferramentas de Suporte

- **Git**: Controlo de versão e rastreabilidade
- **VS Code**: Desenvolvimento e debugging
- **Jupyter Notebook** (opcional): Análise exploratória

---

## Referências Bibliográficas

1. Zhang, Q., & Li, H. (2007). "MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition". *IEEE Transactions on Evolutionary Computation*, 11(6), 712-731.

2. Zitzler, E., & Thiele, L. (1999). "Multiobjective Evolutionary Algorithms: A Comparative Case Study and the Strength Pareto Approach". *IEEE Transactions on Evolutionary Computation*, 3(4), 257-271.

3. Harris, C. R., et al. (2020). "Array programming with NumPy". *Nature*, 585(7825), 357-362.

4. Google. (2005). "General Transit Feed Specification (GTFS)". Retrieved from https://developers.google.com/transit/gtfs

5. UITP. (2017). "Passenger Perception of Intermodal Transfers in Public Transport". *Urban Transit Professionals*, Brussels.

6. Hagberg, A., Schult, D., & Swart, P. (2008). "Exploring network structure, dynamics, and function using NetworkX". In *Proceedings of the 7th Python in Science Conference*, 11-15.