# TRABALHO 02 — Análise Computational de Problemas de Scheduling

**Aluno:** Regivaldo Araújo  
**Disciplina:** Otimização 2026  
**Programa:** PPGEE/UFAM  
**Professor:** Kenny Vinente dos Santos  
**Prazo:** 09/07/2026  
**Entrega:** Pull Request no repositório da disciplina

---

## Resumo

Este trabalho implementa e analisa computacionalmente três formulações MILP (Mixed-Integer Linear Programming) disjuntivas (big-M) para problemas clássicos de escalonamento, utilizando **Julia 1.12** com o framework **JuMP** e o solver **HiGHS**. São avaliadas 48 instâncias distribuídas em três classes:

| Classe | Notação | Instâncias | Ótimos Encontrados |
|--------|---------|:----------:|:-----------------:|
| Machine Scheduling | 1 \| r_j \| ΣT_j | 31 | varia |
| Job Shop | J ‖ C_max | 10 | varia |
| Flow Shop (permutação) | F \| prmu \| C_max | 7 | varia |

---

## Problemas e Formulações

### 1. Machine Scheduling — `1 | r_j | ΣT_j`

Máquina única com datas de liberação `r_j`. Minimiza o tardiness total.

```
min  Σ T_j
s.t. s_j ≥ r_j                          ∀j   (liberação)
     T_j ≥ s_j + p_j − d_j, T_j ≥ 0    ∀j   (tardiness)
     s_j ≥ s_i + p_i − H(1−x_ij)       ∀i<j (disjunção)
     s_i ≥ s_j + p_j − H·x_ij          ∀i<j
     x_ij ∈ {0,1}
```

**Complexidade do modelo:** O(n²) variáveis binárias de precedência.

**Instâncias:** 31 (inst_book n=7, e conjuntos n∈{5,7,9,11,13,15,17,19,21,24}, 3 sementes cada).

---

### 2. Job Shop Scheduling — `J ‖ C_max`

Jobs com roteiros fixos em máquinas dedicadas. Minimiza o makespan.

```
min  Cmax
s.t. s_{j,o+1} ≥ s_{j,o} + p_{j,o}              ∀j,o  (precedência no job)
     Cmax ≥ s_{j,last} + p_{j,last}              ∀j
     s_{j2,o2} ≥ s_{j1,o1} + p_{j1,o1} − M(1−z)  ∀ pares na mesma máquina
     s_{j1,o1} ≥ s_{j2,o2} + p_{j2,o2} − M·z
     z ∈ {0,1}
```

**Instâncias:** 10 da biblioteca JSPLIB (abz5, abz6, ft06, ft10, ft20, la01–la04, orb01).

---

### 3. Flow Shop de Permutação — `F | prmu | C_max`

Mesma sequência de jobs em todas as máquinas. Minimiza o makespan.

```
min  Cmax
s.t. C_{j,1} ≥ p_{j,1}                          ∀j
     C_{j,k} ≥ C_{j,k-1} + p_{j,k}              ∀j,k  (sequência de máquinas)
     C_{j,k} ≥ C_{i,k} + p_{j,k} − M(1−x_ij)   ∀i<j, ∀k
     C_{i,k} ≥ C_{j,k} + p_{i,k} − M·x_ij       ∀i<j, ∀k
     Cmax ≥ C_{j,m}                              ∀j
     x_ij ∈ {0,1}
```

**Instâncias:** 7 arquivos CSV com dimensões 3m×10j a 10m×20j.

---

## Configuração do Solver

```julia
optimizer_with_attributes(HiGHS.Optimizer,
    "time_limit"                       => 30.0,  # 30s Machine / 60s JS / 60s FS
    "mip_rel_gap"                      => 1e-4,
    "primal_feasibility_tolerance"     => 1e-6,
    "dual_feasibility_tolerance"       => 1e-6,
    "output_flag"                      => false,
)
```

---

## Como Reproduzir

### Pré-requisitos

- [Julia ≥ 1.9](https://julialang.org/downloads/)
- Python ≥ 3.8 (apenas para gerar o dashboard)

### Passo a passo

```bash
# 1. Clonar o repositório (ou fork)
git clone https://github.com/<usuario>/optimization2026.git
cd optimization2026/atividades/TRABALHO02

# 2. Instalar dependências Julia
julia julia/install.jl

# 3. (Opcional) Baixar dados originais
python download_data.py

# 4. Executar os três solvers
julia julia/scheduling.jl
# → gera results/results.json

# 5. Gerar o dashboard HTML
python generate_dashboard.py
# → gera dashboard.html

# 6. Abrir o dashboard no navegador
start dashboard.html  # Windows
open dashboard.html   # macOS
```

---

## Estrutura do Repositório

```
TRABALHO02/
├── README.md                  ← este arquivo
├── julia/
│   ├── scheduling.jl          ← implementação MILP completa (3 problemas)
│   ├── Project.toml           ← dependências Julia
│   ├── Manifest.toml          ← versões fixadas
│   └── install.jl             ← script de instalação
├── data/
│   ├── machine/               ← 31 instâncias JSON (1|r_j|ΣT_j)
│   │   └── _index.csv         ← índice das instâncias
│   ├── jobshop/               ← 10 instâncias JSPLIB + metadata
│   │   ├── instances.json
│   │   └── instances/
│   └── flowshop/              ← 7 instâncias CSV
│       └── LICENSE_dataset.txt
├── results/
│   └── results.json           ← saída completa do solver (schedules + gaps)
├── dashboard.html             ← dashboard interativo gerado
├── generate_dashboard.py      ← gerador do dashboard
└── download_data.py           ← script para baixar dados do repositório
```

---

## Resultados Principais

Os resultados completos estão em `results/results.json` e visualizados interativamente em `dashboard.html`.

### Machine Scheduling
- **31 instâncias** testadas com n ∈ {5, 7, 9, 11, 13, 15, 17, 19, 21, 24}
- Solver encontra o ótimo para instâncias pequenas (n ≤ ~11) dentro de 30s
- Para n ≥ 13 o gap MIP cresce rapidamente (relaxação LP fraca com big-M)
- Instância inst_book (n=7, livro MO-book 3.5): **ΣT_j = 16** (ótimo)

### Job Shop
- **10 instâncias JSPLIB** com ótimos conhecidos para comparação
- ft06 (6×6): solucionado no ótimo (Cmax = 55)
- Instâncias 10×10 excedem o tempo limite de 60s
- Desvio médio do ótimo conhecido: variável por instância

### Flow Shop
- **7 instâncias** de 3m×10j a 10m×20j
- Restrição de permutação torna o modelo denso O(J²·M)
- Gap médio: computado via solver HiGHS

---

## Dependências

| Pacote | Versão | Uso |
|--------|--------|-----|
| Julia  | 1.12.6 | Linguagem |
| JuMP   | —      | Modelagem MILP |
| HiGHS  | —      | Solver MILP |
| JSON   | —      | I/O de dados |
| Chart.js | 4.4.1 | Gráficos no dashboard |
| chartjs-plugin-zoom | 2.0.1 | Zoom interativo |

---

## Referências

1. Guéret, C., Prins, C., & Sevaux, M. (2000). *Applications of Optimization with Xpress-MP.*
2. Lawler et al. (1993). *Sequencing and scheduling: Algorithms and complexity.* Handbooks OR&MS, 4.
3. Manne, A. S. (1960). *On the Job-Shop Scheduling Problem.* Operations Research, 8(2).
4. Applegate, D., & Cook, W. (1991). *A Computational Study of the Job-Shop Scheduling Problem.* ORSA JoC, 3(2).
5. Garey, M. R., Johnson, D. S., & Sethi, R. (1976). *The Complexity of Flowshop and Jobshop Scheduling.* MOR, 1(2).
6. Huangfu, Q., & Hall, J. A. J. (2018). *Parallelizing the dual revised simplex method.* MPC, 10(1).
7. Dunning, I., Huchette, J., & Lubin, M. (2017). *JuMP: A Modeling Language for Mathematical Optimization.* SIAM Review, 59(2).
8. Bezanson et al. (2017). *Julia: A fresh approach to numerical computing.* SIAM Review, 59(1).
