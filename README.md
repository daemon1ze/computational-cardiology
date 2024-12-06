# Cardiac Electrophysiology Simulator

A detailed computational model simulating cardiac action potentials and ECG generation with a focus on amphetamine effects on cardiac electrophysiology.

## Overview

This simulator implements a comprehensive cardiac cell model that includes:
- Multiple ion channel currents (Na⁺, Ca²⁺, K⁺)
- HCN ("funny") current
- Na⁺/K⁺ pump and Na⁺/Ca²⁺ exchanger
- Sarcoplasmic reticulum Ca²⁺ dynamics
- ECG generation
- Amphetamine modulation effects

## Features

- **Detailed Ion Channel Modeling**: Implements Hodgkin-Huxley type kinetics for major cardiac ion channels
- **Amphetamine Effects**: Models both acute and chronic effects of amphetamines on:
  - L-type Ca²⁺ channels
  - K⁺ channels
  - HCN channels
  - Ion pumps and exchangers
- **ECG Generation**: Produces realistic ECG waveforms with P-QRS-T morphology
- **Visualization Tools**: Comprehensive plotting functions for:
  - Action potentials
  - Channel gating variables
  - ECG traces
  - Comparative analysis across different amphetamine levels
- **Model Validation**: Automatic validation against physiological parameters

## Requirements

- Python 3.x
- NumPy
- SciPy
- Matplotlib
- Seaborn

## Installation
