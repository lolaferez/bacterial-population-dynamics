# Bacterial Population Dynamics and Growth Strategies

Computational modelling project developed for the **Complex Systems Physics** course of the Double Degree in Physics and Mathematics at the University of Granada.

The project investigates how bacterial populations allocate limited resources between growth and nutrient acquisition, and how different metabolic strategies respond to changing nutritional environments.

The study is based on a simplified **Resource Allocation Model (RAM)** for *E. coli* and combines mathematical modelling, numerical simulation and parameter analysis.

## Project Overview

Bacterial cells operate with limited resources and must distribute them between two main sectors:

- **Catabolic sector (C):** nutrient uptake and processing
- **Ribosomal sector (R):** biomass synthesis and cellular growth

Since the total resource allocation is constrained by

`R + C = 1`

increasing investment in growth necessarily reduces the resources available for nutrient acquisition.

The model describes this trade-off through a nonlinear dynamical system involving intracellular substrate concentration, ribosomal allocation and bacterial growth rate.

## Growth Strategies

Different metabolic strategies are characterised through the ribosomal saturation parameter `k`.

Three representative regimes were studied:

- **Saturated strategy (`k << 1`)**  
  High growth rates in favourable environments, but limited flexibility when environmental conditions change.

- **Sub-saturated strategy (`k ≈ 1`)**  
  Intermediate growth combined with greater capacity to reorganise ribosomal resources.

- **Extremely sub-saturated strategy (`k >> 1`)**  
  Low growth rates but reduced ribosomal reallocation.

This allows the model to investigate the trade-off between maximising instantaneous growth and maintaining adaptability to environmental perturbations.

## Numerical Modelling

The nonlinear differential equations were implemented and solved numerically in **Python**.

The computational workflow includes:

- Implementation of the Resource Allocation Model
- Numerical integration of coupled nonlinear differential equations
- Fourth-order Runge–Kutta (RK4) integration
- Simulation of stationary states
- Nutritional upshift experiments
- Parameter sweeps over metabolic strategies and nutrient availability
- Population competition simulations
- Analysis of ribosomal reallocation
- Growth/adaptation trade-off analysis
- Data visualisation with Matplotlib
- Animated parameter sweeps

## Nutritional Perturbations

The response of the system was studied under changing environmental conditions.

### Nutritional upshift

A sudden increase in nutrient availability drives the system away from its stationary state.

The intracellular substrate responds rapidly, increasing ribosomal activity and growth, while the ribosomal fraction adapts on a slower timescale.

Different metabolic strategies therefore exhibit different transient responses after the same environmental perturbation.

### Gradual nutrient depletion

The model was also studied under progressively deteriorating nutritional conditions.

Saturated strategies initially maintain higher growth rates but experience a sharper decline once nutrient availability becomes limiting, while sub-saturated strategies exhibit a more gradual response.

## Population Dynamics

The Resource Allocation Model was extended to include population abundance and competition between different metabolic strategies.

A logistic population model was coupled to the intracellular dynamics, allowing the simultaneous evolution of:

- ribosomal allocation `R(t)`
- intracellular substrate `x(t)`
- growth rate `μ(t)`
- population abundance `N(t)`
- nutrient availability `β(t)`

This makes it possible to compare the competitive behaviour of different bacterial strategies following environmental changes.

## Parameter Exploration

Parameter sweeps were performed across metabolic strategies and nutritional upshift intensities.

For each configuration, quantities such as

- maximum transient growth rate
- ribosomal reallocation
- population response

were evaluated.

The resulting parameter maps reveal a trade-off between **growth performance and adaptability**: strategies capable of reaching higher growth peaks generally require larger internal resource reorganisation.

## Main Results

The simulations show that no single strategy simultaneously maximises growth and minimises adaptation costs.

The saturated regime achieves high growth rates and can dominate under sufficiently favourable nutritional conditions, but requires substantial ribosomal reallocation.

The extremely sub-saturated regime requires less internal reorganisation but its low growth rate limits its competitive performance.

Within the parameter range investigated, the **sub-saturated strategy provides the best compromise between transient growth and adaptability**, maintaining competitive growth while requiring less ribosomal reorganisation.

## Technologies

- **Python**
- **NumPy**
- **Matplotlib**
- Numerical integration
- Fourth-order Runge–Kutta methods
- Nonlinear dynamical systems
- Parameter sweeps
- Mathematical modelling
- Scientific visualisation

## Repository Structure

```text
.
├── code/
│   ├── population_dynamics.py
│   ├── upshift_population_comparison.py
│   ├── growth_adaptation_heatmaps.py
│   ├── parameter_sweep.py
│   └── nutrient_sweep_animation.py
│
├── plots/
├── report/
├── presentation/
└── README.md