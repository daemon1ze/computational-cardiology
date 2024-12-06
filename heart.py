import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple
import seaborn as sns
from scipy.signal import find_peaks

@dataclass
class ModelParameters:
    """Parameters for the cardiac cell model"""
    Cm: float = 1.0  # Membrane capacitance (uF/cm^2)
    gNa: float = 16.0  # Sodium channel conductance (mS/cm^2)
    gCaL: float = 0.005  # L-type Calcium channel conductance (mS/cm^2)
    gK: float = 0.36  # Potassium channel conductance (mS/cm^2)
    E_Na: float = 50.0  # Sodium reversal potential (mV)
    E_Ca: float = 60.0  # Calcium reversal potential (mV)
    E_K: float = -85.0  # Potassium reversal potential (mV)
    amphetamine_level: float = 1.0  # Amphetamine effect level
    gf: float = 0.1  # HCN channel conductance (mS/cm^2)
    Ef: float = -20.0  # HCN reversal potential (mV)
    gNaK: float = 1.0  # Na/K pump strength
    gNCX: float = 0.5  # Na/Ca exchanger strength

@dataclass
class ECGParameters:
    """Parameters for ECG generation"""
    electrode_distance: float = 10.0  # Distance from heart to electrode (cm)
    conduction_velocity: float = 0.5  # Tissue conduction velocity (m/s)
    qrs_width: float = 0.08  # QRS complex width (s)
    t_wave_factor: float = 0.3  # T wave amplitude factor
    p_wave_factor: float = 0.2  # P wave amplitude factor

class CardiacModel:
    def __init__(self, params: ModelParameters):
        self.params = params
        
    def I_Na(self, V: float, m: float, h: float, j: float) -> float:
        """Sodium current."""
        return self.params.gNa * (m**3) * h * j * (V - self.params.E_Na)
    
    def I_CaL(self, V: float, d: float, f: float) -> float:
        """L-type calcium current with amphetamine and β1-adrenergic effects."""
        # Amphetamines increase Ca2+ current both directly and via β1 pathway
        beta1_effect = 0.5 * self.params.amphetamine_level  # β1-mediated phosphorylation
        direct_effect = 0.2 * self.params.amphetamine_level  # Direct channel modification
        gCaL_amphetamine = self.params.gCaL * (1 + beta1_effect + direct_effect)
        return gCaL_amphetamine * d * f * (V - self.params.E_Ca)
    
    def I_K(self, V: float, n: float) -> float:
        """Potassium current with amphetamine effect."""
        gK_amphetamine = self.params.gK * (1 - 0.1 * self.params.amphetamine_level)
        return gK_amphetamine * (n**4) * (V - self.params.E_K)
    
    def I_f(self, V: float, y: float) -> float:
        """Funny current (HCN) with amphetamine effect."""
        # Amphetamines increase If via cAMP
        gf_amphetamine = self.params.gf * (1 + 0.3 * self.params.amphetamine_level)
        return gf_amphetamine * y * (V - self.params.Ef)
    
    def I_NaK(self, V: float, Nai: float) -> float:
        """Na/K pump current with chronic amphetamine effects."""
        # Chronic exposure can reduce pump function
        chronic_effect = max(0, 1 - 0.2 * self.params.amphetamine_level)
        return self.params.gNaK * chronic_effect * Nai / (Nai + 10)
    
    def I_NCX(self, V: float, Nai: float, Cai: float) -> float:
        """Na/Ca exchanger current."""
        # Similar chronic effects on NCX
        chronic_effect = max(0, 1 - 0.15 * self.params.amphetamine_level)
        return self.params.gNCX * chronic_effect * (Nai/Cai - np.exp(-V/60))
    
    def I_total(self, V: float, m: float, h: float, j: float, d: float, f: float, n: float, y: float, Nai: float, Cai: float) -> float:
        """Total ionic current."""
        return self.I_Na(V, m, h, j) + self.I_CaL(V, d, f) + self.I_K(V, n) + self.I_f(V, y) + self.I_NaK(V, Nai) + self.I_NCX(V, Nai, Cai)

    @staticmethod
    def alpha_beta_functions(V: float) -> dict:
        """Calculate all alpha and beta rate constants."""
        return {
            'alpha_m': 0.1 * (25 - V) / (np.exp((25 - V) / 10) - 1),
            'beta_m': 4.0 * np.exp(-V / 18),
            'alpha_h': 0.07 * np.exp(-V / 20),
            'beta_h': 1.0 / (np.exp((30 - V) / 10) + 1),
            'alpha_j': 0.07 * np.exp(-V / 20),
            'beta_j': 1.0 / (np.exp((30 - V) / 10) + 1),
            'alpha_d': 0.095 * np.exp(-(V - 5) / 100) / (1 + np.exp((V - 5) / 13)),
            'beta_d': 0.07 * np.exp(-(V + 44) / 59) / (1 + np.exp(-(V + 44) / 20)),
            'alpha_f': 0.012 * np.exp(-(V + 28) / 125) / (1 + np.exp((V + 28) / 6.5)),
            'beta_f': 0.0065 * np.exp(-(V + 30) / 50) / (1 + np.exp(-(V + 30) / 5)),
            'alpha_n': 0.01 * (10 - V) / (np.exp((10 - V) / 10) - 1),
            'beta_n': 0.125 * np.exp(-V / 80),
            'alpha_y': 0.1 / (1 + np.exp((V + 80) / 5)),  # HCN activation
            'beta_y': 0.1 / (1 + np.exp(-(V + 80) / 5))   # HCN deactivation
        }

    def simulate(self, t: np.ndarray, y0: List[float]) -> Tuple[np.ndarray, dict]:
        """Run the simulation and return results with additional metrics."""
        def deriv(y: List[float], t: float) -> List[float]:
            # Update state variables to include CaSR
            V, m, h, j, d, f, n, y, Nai, Cai, CaSR = y
            rates = self.alpha_beta_functions(V)
            
            dm_dt = rates['alpha_m'] * (1 - m) - rates['beta_m'] * m
            dh_dt = rates['alpha_h'] * (1 - h) - rates['beta_h'] * h
            dj_dt = rates['alpha_j'] * (1 - j) - rates['beta_j'] * j
            dd_dt = rates['alpha_d'] * (1 - d) - rates['beta_d'] * d
            df_dt = rates['alpha_f'] * (1 - f) - rates['beta_f'] * f
            dn_dt = rates['alpha_n'] * (1 - n) - rates['beta_n'] * n
            dy_dt = rates['alpha_y'] * (1 - y) - rates['beta_y'] * y
            
            # Calculate ICaL for calcium dynamics
            ICaL = self.I_CaL(V, d, f)
            dNai_dt, dCai_dt, dCaSR_dt = self.calcium_dynamics(Cai, CaSR, ICaL, V, m, h, j)
            
            I_ion = self.I_total(V, m, h, j, d, f, n, y, Nai, Cai)
            dV_dt = -I_ion / self.params.Cm
            
            return [dV_dt, dm_dt, dh_dt, dj_dt, dd_dt, df_dt, dn_dt, dy_dt, dNai_dt, dCai_dt, dCaSR_dt]
        
        solution = odeint(deriv, y0, t)
        
        # Calculate additional metrics
        metrics = {
            'peak_voltage': np.max(solution[:, 0]),
            'min_voltage': np.min(solution[:, 0]),
            'action_potential_duration': self._calculate_apd(t, solution[:, 0]),
            'upstroke_velocity': np.max(np.diff(solution[:, 0]) / np.diff(t))
        }
        
        return solution, metrics
    
    def _calculate_apd(self, t: np.ndarray, V: np.ndarray, threshold: float = 0.9) -> float:
        """Calculate Action Potential Duration at specified threshold."""
        peak_idx = np.argmax(V)
        peak_value = V[peak_idx]
        resting_value = V[0]
        threshold_value = resting_value + threshold * (peak_value - resting_value)
        
        # Find crossing points
        above_threshold = V > threshold_value
        crossings = np.where(np.diff(above_threshold))[0]
        
        if len(crossings) >= 2:
            return t[crossings[1]] - t[crossings[0]]
        return 0.0

    def calcium_dynamics(self, Cai: float, CaSR: float, ICaL: float, V: float, m: float, h: float, j: float) -> Tuple[float, float, float]:
        """Calculate changes in intracellular and SR calcium."""
        # β1-adrenergic effects enhance SR calcium release
        beta1_effect = 1 + 0.4 * self.params.amphetamine_level
        
        # SR calcium release
        J_rel = beta1_effect * (CaSR - Cai) * 0.15
        
        # SR calcium uptake (SERCA)
        J_up = beta1_effect * Cai * 0.004
        
        dCai_dt = -ICaL + J_rel - J_up
        dCaSR_dt = -J_rel + J_up
        dNai_dt = -self.I_Na(V, m, h, j) / (self.params.Cm * 96485)  # Faraday constant
        
        return dNai_dt, dCai_dt, dCaSR_dt

    def generate_ecg(self, t: np.ndarray, V: np.ndarray, ecg_params: ECGParameters) -> np.ndarray:
        """Generate ECG signal with realistic morphology."""
        ecg = np.zeros_like(t)
        
        # Heart rate increases with amphetamine level (base 75 bpm)
        heart_rate = 75 * (1 + 0.2 * self.params.amphetamine_level)
        beat_interval = int(60000 / heart_rate / (t[1] - t[0]))  # Convert bpm to indices
        
        # Generate beats
        for beat_start in range(0, len(t), beat_interval):
            if beat_start + 200 >= len(t):  # Ensure enough space for full complex
                break
            
            # Time windows (in indices)
            p_start = beat_start
            qrs_start = p_start + 40
            t_end = qrs_start + 160
            
            # P wave
            p_duration = 40
            if p_start + p_duration < len(t):
                p_wave = 0.25 * np.sin(np.linspace(0, np.pi, p_duration))
                ecg[p_start:p_start + p_duration] = p_wave
            
            # QRS complex
            qrs_duration = 40
            if qrs_start + qrs_duration < len(t):
                # Q wave
                q_duration = 10
                ecg[qrs_start:qrs_start + q_duration] = -0.3
                
                # R wave (sharp upstroke)
                r_duration = 20
                r_wave = np.concatenate([
                    np.linspace(0, 1.2, r_duration//2),
                    np.linspace(1.2, 0, r_duration//2)
                ])
                ecg[qrs_start + q_duration:qrs_start + q_duration + r_duration] = r_wave
                
                # S wave
                s_duration = 10
                ecg[qrs_start + q_duration + r_duration:qrs_start + qrs_duration] = -0.2
            
            # ST segment
            st_start = qrs_start + qrs_duration
            st_duration = 40
            if st_start + st_duration < len(t):
                # ST elevation increases with amphetamine
                st_elevation = 0.1 * self.params.amphetamine_level
                ecg[st_start:st_start + st_duration] = st_elevation
            
            # T wave
            t_start = st_start + st_duration
            t_duration = 80
            if t_start + t_duration < len(t):
                # T wave amplitude and width affected by amphetamine
                t_amp = 0.3 * (1 + 0.3 * self.params.amphetamine_level)
                t_wave = t_amp * np.sin(np.linspace(0, np.pi, t_duration))
                ecg[t_start:t_start + t_duration] = t_wave
        
        return ecg

    def validate_model(self, solution: np.ndarray, t: np.ndarray) -> dict:
        """Validate model against known physiological parameters."""
        V = solution[:, 0]
        
        # Calculate key metrics
        resting_potential = np.mean(V[0:int(len(V)/10)])  # First 10% of simulation
        peaks, _ = find_peaks(V, height=0)
        peak_potential = np.mean(V[peaks]) if len(peaks) > 0 else 0
        
        # Calculate APD90
        apd90 = self._calculate_apd(t, V, threshold=0.9)
        
        # Calculate dV/dt max (upstroke velocity)
        dvdt_max = np.max(np.gradient(V, t))
        
        # Define physiological ranges
        validation = {
            'resting_potential': {
                'value': resting_potential,
                'normal_range': (-90, -80),
                'is_normal': -90 <= resting_potential <= -80
            },
            'peak_potential': {
                'value': peak_potential,
                'normal_range': (20, 40),
                'is_normal': 20 <= peak_potential <= 40
            },
            'apd90': {
                'value': apd90,
                'normal_range': (200, 300),
                'is_normal': 200 <= apd90 <= 300
            },
            'upstroke_velocity': {
                'value': dvdt_max,
                'normal_range': (100, 400),
                'is_normal': 100 <= dvdt_max <= 400
            }
        }
        
        return validation

def plot_results(t: np.ndarray, solution: np.ndarray, metrics: dict) -> None:
    """Create comprehensive visualizations of the simulation results."""
    sns.set_style("whitegrid")
    
    fig = plt.figure(figsize=(15, 10))
    gs = plt.GridSpec(3, 2)
    
    # Main action potential plot
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t, solution[:, 0], 'b-', label='Membrane Potential')
    ax1.set_title('Cardiac Action Potential')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Membrane Potential (mV)')
    ax1.legend()
    
    # Gating variables
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t, solution[:, 1], 'r-', label='m (Na+ activation)')
    ax2.plot(t, solution[:, 2], 'g-', label='h (Na+ inactivation)')
    ax2.plot(t, solution[:, 3], 'b-', label='j (Na+ slow inactivation)')
    ax2.set_title('Sodium Channel Gating Variables')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Gate Value')
    ax2.legend()
    
    # Calcium and Potassium gating variables
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(t, solution[:, 4], 'r-', label='d (Ca2+ activation)')
    ax3.plot(t, solution[:, 5], 'g-', label='f (Ca2+ inactivation)')
    ax3.plot(t, solution[:, 6], 'b-', label='n (K+ activation)')
    ax3.set_title('Ca2+ and K+ Channel Gating Variables')
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Gate Value')
    ax3.legend()
    
    # Metrics summary
    ax4 = fig.add_subplot(gs[2, :])
    metrics_text = (
        f"Peak Voltage: {metrics['peak_voltage']:.2f} mV\n"
        f"Minimum Voltage: {metrics['min_voltage']:.2f} mV\n"
        f"Action Potential Duration: {metrics['action_potential_duration']:.2f} ms\n"
        f"Maximum Upstroke Velocity: {metrics['upstroke_velocity']:.2f} mV/ms"
    )
    ax4.text(0.1, 0.5, metrics_text, fontsize=12, va='center')
    ax4.set_axis_off()
    
    plt.tight_layout()
    plt.show()

def plot_comparative_results(t: np.ndarray, solutions: List[np.ndarray], metrics_list: List[dict], 
                           amphetamine_levels: List[float], model: CardiacModel) -> None:
    """Create comparative visualizations including ECG."""
    sns.set_style("whitegrid")
    colors = sns.color_palette("husl", len(solutions))
    
    # Create figure with adjusted height for better ECG visibility
    fig = plt.figure(figsize=(15, 18))  # Increased height
    gs = plt.GridSpec(5, 2, height_ratios=[3, 2, 2, 1.5, 3])  # Increased last ratio for ECG
    
    # Main action potential plot
    ax1 = fig.add_subplot(gs[0, :])
    for i, (solution, level, color) in enumerate(zip(solutions, amphetamine_levels, colors)):
        ax1.plot(t, solution[:, 0], color=color, label=f'Amphetamine Level: {level}')
    ax1.set_title('Cardiac Action Potential Comparison')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Membrane Potential (mV)')
    ax1.legend()
    
    # Calcium current comparison
    ax2 = fig.add_subplot(gs[1, 0])
    for i, (solution, level, color) in enumerate(zip(solutions, amphetamine_levels, colors)):
        d, f = solution[:, 4], solution[:, 5]
        ax2.plot(t, d * f, color=color, label=f'Level: {level}')
    ax2.set_title('Ca2+ Channel Activity')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Gate Product (d*f)')
    ax2.legend()
    
    # Potassium current comparison
    ax3 = fig.add_subplot(gs[1, 1])
    for i, (solution, level, color) in enumerate(zip(solutions, amphetamine_levels, colors)):
        n = solution[:, 6]
        ax3.plot(t, n**4, color=color, label=f'Level: {level}')
    ax3.set_title('K+ Channel Activity')
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Gate Value (n^4)')
    ax3.legend()
    
    # Metrics comparison table
    ax4 = fig.add_subplot(gs[2, :])
    metrics_comparison = {
        'Amphetamine Level': amphetamine_levels,
        'Peak Voltage (mV)': [m['peak_voltage'] for m in metrics_list],
        'Min Voltage (mV)': [m['min_voltage'] for m in metrics_list],
        'APD (ms)': [m['action_potential_duration'] for m in metrics_list],
        'Max Upstroke Velocity (mV/ms)': [m['upstroke_velocity'] for m in metrics_list]
    }
    
    cell_text = []
    for i in range(len(amphetamine_levels)):
        cell_text.append([
            f"{metrics_comparison['Amphetamine Level'][i]:.2f}",
            f"{metrics_comparison['Peak Voltage (mV)'][i]:.2f}",
            f"{metrics_comparison['Min Voltage (mV)'][i]:.2f}",
            f"{metrics_comparison['APD (ms)'][i]:.2f}",
            f"{metrics_comparison['Max Upstroke Velocity (mV/ms)'][i]:.2f}"
        ])
    
    table = ax4.table(cellText=cell_text,
                     colLabels=list(metrics_comparison.keys()),
                     loc='center',
                     cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax4.set_axis_off()
    
    # ECG subplot with improved visualization
    ecg_params = ECGParameters()
    ax_ecg = fig.add_subplot(gs[4, :])
    
    # Show multiple beats
    t_start_idx = 0
    t_end_idx = int(len(t) * 0.8)  # Show more of the time series
    
    for i, (solution, level, color) in enumerate(zip(solutions, amphetamine_levels, colors)):
        ecg = model.generate_ecg(t, solution[:, 0], ecg_params)
        ax_ecg.plot(t[t_start_idx:t_end_idx], 
                   ecg[t_start_idx:t_end_idx], 
                   color=color, 
                   label=f'Amphetamine Level: {level}',
                   linewidth=1.5)
    
    ax_ecg.set_title('Simulated ECG')
    ax_ecg.set_xlabel('Time (ms)')
    ax_ecg.set_ylabel('ECG Amplitude (mV)')
    ax_ecg.legend()
    ax_ecg.set_ylim(-0.5, 1.2)
    ax_ecg.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_ecg_separately(t: np.ndarray, solutions: List[np.ndarray], 
                       amphetamine_levels: List[float], model: CardiacModel) -> None:
    """Plot ECG in a separate figure window."""
    plt.figure('ECG Visualization', figsize=(15, 8))
    colors = sns.color_palette("husl", len(solutions))
    
    # Show multiple beats
    t_start_idx = 0
    t_end_idx = int(len(t) * 0.4)  # Show first 40% of time series
    
    ecg_params = ECGParameters()
    for i, (solution, level, color) in enumerate(zip(solutions, amphetamine_levels, colors)):
        # Generate ECG and add offset for visibility
        ecg = model.generate_ecg(t[:t_end_idx], solution[:t_end_idx, 0], ecg_params)
        offset = i * 1.5  # Increased spacing between traces
        plt.plot(t[:t_end_idx], ecg + offset, 
                color=color, 
                label=f'Amphetamine Level: {level}',
                linewidth=1.5)
    
    plt.title('Simulated ECG')
    plt.xlabel('Time (ms)')
    plt.ylabel('ECG Amplitude (mV)')
    plt.legend(loc='upper right')
    plt.ylim(-0.5, 6.0)  # Adjusted to show all traces
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Force separate window
    plt.figure('ECG Visualization').show()

def main():
    # Define different amphetamine levels to simulate
    amphetamine_levels = [0.0, 0.5, 1.0, 2.0]  # Including baseline (0.0)
    
    # Initial conditions
    # [V, m, h, j, d, f, n, y, Nai, Cai, CaSR]
    y0 = [-85.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 140.0, 100.0, 1000.0]  # Added CaSR initial value
    
    # Time vectorß
    t = np.linspace(0, 1000, 10000)
    
    # Run simulations for each amphetamine level
    solutions = []
    metrics_list = []
    
    for level in amphetamine_levels:
        params = ModelParameters(amphetamine_level=level)
        model = CardiacModel(params)
        solution, metrics = model.simulate(t, y0)
        
        # Add validation
        validation_results = model.validate_model(solution, t)
        print(f"\nValidation Results for Amphetamine Level {level}:")
        for param, results in validation_results.items():
            status = "✓" if results['is_normal'] else "✗"
            print(f"{param}: {results['value']:.2f} {status}")
            print(f"Normal range: {results['normal_range']}")
        
        solutions.append(solution)
        metrics_list.append(metrics)
    
    # Create main figure with a specific name
    plt.figure('Main Plots', figsize=(15, 15))
    # Plot the regular comparative results
    plot_comparative_results(t, solutions, metrics_list, amphetamine_levels, model)
    
    # Plot ECG in separate window
    plot_ecg_separately(t, solutions, amphetamine_levels, model)
    
    plt.show()  # Show all figures

if __name__ == "__main__":
    main()