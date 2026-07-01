# Modelagem Numérica do Acoplamento Hidromecânico em Reservatórios



Este repositório contém o trabalho final desenvolvido para a disciplina **GA-020 – Solução Numérica de Equações Diferenciais**, do Programa de **Programa de Pós-Graduação de Modelagem Computacional (LNCC)**.



---



# Objetivo do Trabalho



O objetivo deste projeto é desenvolver um modelo computacional para simular o comportamento hidromecânico de um reservatório poroso saturado utilizando o **Método dos Elementos Finitos (MEF)** através do framework **Firedrake**.



O modelo considera a teoria da **poroelasticidade linear de Biot** sob a hipótese de **acoplamento unidirecional (One-Way Coupling)**, em que:



1. Resolve-se a difusão transiente da pressão de poros governada pela Lei de Darcy;

2. O campo de pressão obtido é utilizado como carregamento para determinar a resposta mecânica do meio poroso através da elasticidade linear.



Ao final da simulação são obtidas diversas quantidades de interesse de engenharia, como:



- Evolução temporal da pressão média;

- Deslocamento máximo;

- Energia elástica;

- Campo de pressão;

- Campo de deslocamentos;

- Velocidade de Darcy;

- Deformação volumétrica;

- Tensão equivalente de von Mises;

- Perfis hidromecânicos ao longo do domínio.



---



# Estrutura do Repositório



```text

PROJETO-HIDROMECANICA/

│

├── notebooks/

│   └── oneway.ipynb

│

├── outputs/

│   ├── Fluxo_Mecanica.pdf

│   ├── campos_P_U.pdf

│   ├── campoU_quiver.pdf

│   ├── Ux_Uy_U.pdf

│   ├── Vel_Darcy.pdf

│   ├── DefVol_VonMises.pdf

│   └── Perfis_Ly_2.pdf

│

├── pixi.toml

├── pixi.lock

├── README.md

└── LICENSE

```
---



# Como Reproduzir os Resultados



Siga os passos abaixo para configurar o ambiente e reproduzir todos os resultados apresentados.



## Pré-requisitos



- Git

- Pixi

- Firedrake



Este projeto utiliza o framework **Firedrake** para a resolução numérica do problema de Elementos Finitos.



A instalação do Firedrake pode ser realizada conforme a documentação oficial:



https://www.firedrakeproject.org/install.html



> **Observação:** Este trabalho foi desenvolvido em um computador com Windows utilizando o **Windows Subsystem for Linux (WSL2)**. Entretanto, o código pode ser ser executado em qualquer sistema operacional suportado pelo Firedrake, desde que o ambiente seja configurado corretamente.



---



# 1. Clonar o Repositório




```bash
git clone [https://github.com/lncc-ga020/trabalho-final-ga020-2026-AlexandreAMoreira.git](https://github.com/lncc-ga020/trabalho-final-ga020-2026-AlexandreAMoreira.git)
cd trabalho-final-ga020-2026-AlexandreAMoreira

```



---



# 2. Configurar o Ambiente



Este projeto utiliza o gerenciador de ambientes **Pixi**.



Caso ainda não possua o Pixi instalado, consulte:



https://pixi.sh/latest/



Após instalar o Pixi, execute



```bash

pixi install

```



Em seguida, ative o ambiente



```bash

pixi shell

```



Todas as dependências do projeto serão instaladas automaticamente.



---



# 3. Executar o Notebook



Com o ambiente ativado, execute



```bash

jupyter lab

```



ou



```bash

jupyter notebook

```



Abra o notebook



```text

notebooks/oneway.ipynb

```



e execute todas as células na ordem apresentada.



---



# Resultados Esperados



Ao término da execução será criada automaticamente a pasta



```text

outputs/

```



contendo as seguintes figuras:



| Arquivo | Descrição |

|---------|-----------|

| Fluxo_Mecanica.pdf | Evolução temporal da pressão média, deslocamento máximo e energia elástica |

| campos_P_U.pdf | Evolução espacial da pressão e da magnitude dos deslocamentos |

| campoU_quiver.pdf | Campo vetorial de deslocamentos |

| Ux_Uy_U.pdf | Magnitude e componentes do deslocamento |

| Vel_Darcy.pdf | Campo de velocidade de Darcy |

| DefVol_VonMises.pdf | Deformação volumétrica e tensão equivalente de von Mises |

| Perfis_Ly_2.pdf | Perfis hidromecânicos ao longo da linha central do domínio |



Além das figuras, o código imprime no terminal algumas quantidades de interesse de engenharia, incluindo:



- Pressão média do reservatório;

- Limites do deslocamento horizontal;

- Limites do deslocamento vertical;

- Magnitude máxima do deslocamento;

- Energia elástica total.



---



# Modelo Matemático

O problema considera um modelo de **poroelasticidade linear** baseado na teoria de **Biot** com **acoplamento unidirecional (One-Way Coupling)**.

O sistema resolvido é composto por:

- Equação de difusão da pressão de poros, governada pela Lei de Darcy;

- Equação de equilíbrio da elasticidade linear isotrópica.

A discretização temporal é realizada através do método de **Euler Implícito**, enquanto a discretização espacial utiliza o **Método dos Elementos Finitos** implementado no Firedrake.



---



# Dependências



As principais bibliotecas utilizadas são:



- Firedrake

- PETSc

- NumPy

- Matplotlib



As dependências do projeto são gerenciadas automaticamente pelo **Pixi**.



---



# Autor



**Alexandre Altamir Moreira - Mestrando**

Programa de Pós Graduação em Modelagem Computacional
Laboratório Nacional de Computação Científica (LNCC)
2026

---


# Licença

Este projeto está licenciado sob a licença MIT. Consulte o arquivo `LICENSE` para mais informações.

---

## Declaração de Uso de IA

A revisão estrutural, refatoração e automação dos scripts deste repositório foram assistidas por ferramentas de Inteligência Artificial Generativa.
* **Modelos utilizados:** Gemini (Google), ChatGPT.

---

## 🤝 Apoio Institucional

Este projeto recebe apoio institucional do [Laboratório Nacional de Computação Científica (LNCC)](https://www.gov.br/lncc/pt-br).
<p align="left">
  <img src="resources/logo/lncc-mcti.svg" alt="Logo LNCC MCTI" width="250px"/>
</p>
