# Instruções de Implementação para o Notebook de Jogos Estocásticos

**Objetivo:** Restaurar e finalizar o notebook do projeto utilizando o arquivo original como base (`team_game_vs_non_cooperative_simple_stochastic_game_original.ipynb`), implementando melhorias estruturais para evitar repetição de código e cobrir os requisitos das Partes 3 e 4 do trabalho (restrição de backlog e análise paramétrica).

**Regra de Ouro:** Não apague as células originais do notebook base. Adicione as novas implementações em uma nova seção ao final do notebook, chamada "Parte 3 & 4: Análise Avançada e Restrições".

---

### 1. Modificação da Função Solver Principal (`queuestr_lp`)
Precisamos atualizar a função de programação linear existente para suportar a restrição de backlog sem duplicar código.

* **Ação:** Localize a função `queuestr_lp` original.
* **Alteração:** Modifique a assinatura da função para aceitar um argumento opcional novo: `Bmax` (default = `None`).
* **Lógica Interna:**
    * Mantenha toda a lógica original de fluxo e conservação de probabilidade.
    * Adicione uma condicional: **Se** `Bmax` for um número (não `None`):
        1.  Calcule os coeficientes para o backlog médio (multiplicação do estado da fila pela probabilidade daquele estado).
        2.  Adicione uma nova linha na matriz de desigualdade (`A_ub` e `b_ub`) do `linprog`.
        3.  A restrição deve representar: $\sum (TamanhoFila \times Probabilidade) \le B_{max}$.

---

### 2. Criação do "Motor de Experimentos" (Grid Search Engine)
Em vez de loops manuais repetidos, crie uma função genérica para rodar baterias de testes.

* **Nome Sugerido:** `run_experiment_batch`
* **Entrada:** Recebe um dicionário onde as chaves são os nomes dos parâmetros (ex: `ArrProb`, `v1`, `Mode`) e os valores são **listas** de valores a testar.
* **Lógica:**
    1.  Gere o produto cartesiano de todas as listas de parâmetros (todas as combinações possíveis).
    2.  Itere sobre cada combinação.
    3.  Configure o objeto `Settings` com os parâmetros da combinação atual.
    4.  Execute a função `queuestr_lp` (já atualizada com Bmax) para o modo especificado (Cooperativo ou Não-Cooperativo).
    5.  Armazene os resultados (Throughput, Backlog Médio, e as Matrizes de Política Resultantes) em uma lista de dicionários/registros.
* **Saída:** Uma lista/DataFrame contendo os resultados de todos os cenários testados.

---

### 3. Implementação da Parte 3 (Restrição de Backlog e Heatmaps)
Objetivo: Reproduzir a análise visual de políticas de admissão sob restrição.

* **Execução:** Use o "Motor de Experimentos" para rodar dois cenários específicos (mantendo outros parâmetros padrão):
    1.  Cenário Rigoroso: `Bmax = 1.0`
    2.  Cenário Relaxado: `Bmax = 4.0`
* **Plotagem:** Crie uma função de visualização que receba esses resultados e gere **Heatmaps** (Mapas de Calor) para:
    * Política de Transmissão (Potência).
    * Política de Admissão (Probabilidade de aceitar pacote).
* **Foco:** Os gráficos devem evidenciar a "região de rejeição" (onde a admissão é zero) quando o buffer está crítico ou o canal é ruim.

---

### 4. Implementação da Parte 4 (Análise Paramétrica - Ceteris Paribus)
Objetivo: Gerar curvas de desempenho variando um parâmetro por vez, conforme exigido no PDF. Use o "Motor de Experimentos" para rodar as 4 baterias abaixo e plote gráficos de linha (Eixo X = Parâmetro, Eixo Y = Throughput/Backlog, Linhas = Coop vs Não-Coop).

**Bateria A: Impacto da Carga (`ArrProb`)**
* **Varie:** `ArrProb` em `[0.2, 0.4, 0.6, 0.8]`.
* **Fixo:** `Bmax=3.0`, Buffer Físico=6.
* **Objetivo:** Demonstrar saturação do sistema.

**Bateria B: Impacto da Potência e Interferência (`v1`, `v2`)**
* **Varie:** Potência (`v1` e `v2` simultaneamente) em `[1.0, 2.0, 3.0, 5.0]`.
* **Fixo:** `Bmax=3.0`, `ArrProb=0.5`.
* **Objetivo:** Discutir como o aumento de potência no modo Não-Cooperativo gera interferência e retornos decrescentes, diferentemente do modo Cooperativo.

**Bateria C: Impacto da Granularidade do Canal (`NLinkStates`)**
* **Varie:** `NLinkStates` em `[2, 4, 6]`.
* **Fixo:** `Bmax=3.0`.
* **Objetivo:** Verificar sensibilidade do modelo à resolução do canal.

**Bateria D: Impacto do Buffer Físico (`NBufferStates`)**
* **Varie:** `NBufferStates` em `[4, 6, 8, 10]`.
* **Fixo:** `Bmax=3.0`, `ArrProb=0.6`.
* **Objetivo:** Demonstrar que aumentar o buffer físico não altera o desempenho quando a restrição de backlog médio ($B_{max}$) é ativa e dominante (o gráfico deve ser plano para N>6).

---

### Resumo da Entrega de Código
O código final deve conter:
1.  A classe `Settings` original.
2.  A função `queuestr_lp` atualizada (com lógica de Bmax).
3.  A função `run_experiment_batch`.
4.  Funções de plotagem (`plot_results_line`, `plot_policy_heatmap`).
5.  As chamadas para gerar as figuras da Parte 3 e Parte 4 descritas acima.