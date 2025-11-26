"""
Flask backend API for Stress-Strain Analysis
Supports TWO approaches:
1. Mathematical/Computational (default) - No ML
2. Machine Learning - Using Linear Regression

Mathematical analysis inspired by:
https://github.com/tomwalton78/Stress-Strain-Curve-Analyser (MIT License)

Features:
- Smooth curve using Savitzky-Golay filter
- R-squared based elastic region detection
- 0.2% offset yield point via line intersection (brute force)
- UTS, % Elongation, Resilience, Toughness
- Region marking (Elastic, Plastic, Necking)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import io
import base64
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.integrate import trapezoid
from scipy.interpolate import UnivariateSpline
import scipy.stats
from material_library import get_material_list, get_material_data, get_categories

app = Flask(__name__)
CORS(app)

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Dual-Mode Backend Running (Math + ML)'})


# ============================================================================
# MATERIAL LIBRARY ENDPOINTS
# ============================================================================
@app.route('/api/materials', methods=['GET'])
def list_materials():
    """Get list of all available materials"""
    try:
        materials = get_material_list()
        categories = get_categories()
        return jsonify({
            'materials': materials,
            'categories': categories
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/materials/<material_id>', methods=['GET'])
def get_material(material_id):
    """Get stress-strain data for a specific material"""
    try:
        data = get_material_data(material_id)
        if data is None:
            return jsonify({'error': 'Material not found'}), 404
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze-material', methods=['POST'])
def analyze_material():
    """Analyze a material from the library"""
    try:
        data = request.get_json()
        material_id = data.get('materialId')
        analysis_method = data.get('analysisMethod', 'mathematical')
        decimal_places = int(data.get('outputDecimalPlaces', 3))
        smoothing_window = int(data.get('smoothingWindow', 11))
        smoothing_order = int(data.get('smoothingOrder', 3))
        
        # Get material data
        mat_data = get_material_data(material_id)
        if mat_data is None:
            return jsonify({'error': 'Material not found'}), 404
        
        # Create DataFrame
        df = pd.DataFrame({
            'Strain': mat_data['strain'],
            'Stress': mat_data['stress']
        })
        
        # Analyze
        if analysis_method.lower() == 'ml':
            results = analyze_ml(df, decimal_places)
        else:
            results = analyze_mathematical(df, decimal_places, smoothing_window, smoothing_order)
        
        # Add material info to results
        results['materialName'] = mat_data['name']
        results['materialCategory'] = mat_data['category']
        results['materialDescription'] = mat_data['description']
        
        return jsonify(results)
    
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# ============================================================================
# MATHEMATICAL APPROACH (Optimized - inspired by tomwalton78 algorithms)
# ============================================================================

def analyze_mathematical(df, decimal_places, smoothing_window, smoothing_order):
    """
    Pure mathematical/computational analysis - OPTIMIZED VERSION
    
    Algorithms inspired by: https://github.com/tomwalton78/Stress-Strain-Curve-Analyser (MIT License)
    But optimized for speed using numpy vectorization.
    
    Special handling for elastomers (strain > 50%):
    - Skips 0.2% offset yield (not applicable)
    - Uses tangent modulus for E
    - Uses proportional limit instead of yield point
    """
    
    strain_raw = df['Strain'].values.astype(float)
    stress_raw = df['Stress'].values.astype(float)
    
    # DETECT MATERIAL TYPE
    max_strain = float(strain_raw.max())
    max_stress = float(stress_raw.max())
    is_elastomer = max_strain > 0.5 and max_stress < 100  # >50% elongation, <100 MPa
    
    # STEP 1: SMOOTH THE DATA
    if len(strain_raw) < smoothing_window:
        smoothing_window = max(5, len(strain_raw) // 4)
        if smoothing_window % 2 == 0:
            smoothing_window += 1
    
    stress_smooth = savgol_filter(stress_raw, smoothing_window, min(smoothing_order, smoothing_window-1))
    
    # STEP 2: FIND ELASTIC MODULUS - Detect where curve deviates from linearity
    def find_elastic_region(strain, stress, is_rubber=False):
        """
        Find the elastic region by detecting where the curve starts to deviate 
        from the initial linear portion. Uses derivative analysis.
        
        CRITICAL: For metals, elastic region is typically strain < 0.002-0.003.
        For polymers/rubbers, it can be larger. We need to detect this carefully.
        """
        n = len(strain)
        
        # APPROACH: Calculate E from the very first points (which are definitely elastic)
        # Then extend outward until we see significant deviation
        
        # Skip the very first point (often noisy or zero)
        start_idx = 1 if strain[0] == 0 else 0
        
        # For initial E estimate, use only the first 3-5 data points
        # These are almost certainly in the elastic region for any material
        initial_end = min(start_idx + 3, n - 1)
        
        # Ensure we have at least 2 points for regression
        if initial_end - start_idx < 2:
            initial_end = min(start_idx + 2, n - 1)
        
        # Get initial slope estimate from earliest points
        initial_slope, initial_intercept, _, _, _ = scipy.stats.linregress(
            strain[start_idx:initial_end+1], stress[start_idx:initial_end+1]
        )
        
        # If initial slope is too low or negative (bad data), try origin-constrained fit
        if initial_slope <= 0:
            # Force through origin: E = sum(stress*strain) / sum(strain^2)
            mask = strain > 0
            if np.any(mask):
                s = strain[mask][:5]
                st = stress[mask][:5]
                initial_slope = np.sum(st * s) / np.sum(s * s)
                initial_intercept = 0
        
        # Now extend the elastic region by checking deviation from this initial line
        predicted_stress = initial_slope * strain + initial_intercept
        
        # Calculate relative deviation at each point
        deviation = np.abs(stress - predicted_stress) / (np.abs(predicted_stress) + 1e-10)
        
        # Find where deviation first exceeds threshold (3% for metals, more conservative)
        elastic_end_idx = initial_end
        deviation_threshold = 0.03  # 3% deviation marks end of elastic region
        
        for i in range(initial_end, n):
            if deviation[i] > deviation_threshold:
                elastic_end_idx = i
                break
            elastic_end_idx = i
        
        # Also check derivative: elastic region has constant slope
        # Calculate point-to-point slopes
        if n > 5:
            slopes = np.diff(stress) / (np.diff(strain) + 1e-10)
            
            # Find where slope drops significantly from initial slope
            for i in range(2, min(elastic_end_idx, len(slopes))):
                if slopes[i] < initial_slope * 0.5:  # Slope dropped to 50% of initial
                    elastic_end_idx = min(elastic_end_idx, i)
                    break
        
        # The elastic modulus should be the initial slope
        # Only re-fit if we have enough points in the truly elastic region
        E = initial_slope
        intercept = initial_intercept
        
        # If elastic_end_idx is larger than start, refit on elastic portion
        if elastic_end_idx > start_idx + 2:
            E, intercept, _, _, _ = scipy.stats.linregress(
                strain[start_idx:elastic_end_idx], stress[start_idx:elastic_end_idx]
            )
            # But use the maximum of this and initial slope (to avoid underestimation)
            if E < initial_slope * 0.8:
                E = initial_slope
                intercept = initial_intercept
        
        return E, intercept, elastic_end_idx
    
    E, intercept, elastic_break_idx = find_elastic_region(strain_raw, stress_smooth)
    
    # Ensure E is positive and reasonable
    if E <= 0:
        end_idx = max(5, len(strain_raw) // 5)
        E, intercept, _, _, _ = scipy.stats.linregress(strain_raw[:end_idx], stress_smooth[:end_idx])
        elastic_break_idx = end_idx
    
    # STEP 3: FIND UTS (Maximum stress)
    uts_idx = np.argmax(stress_smooth)
    sigma_uts = float(stress_smooth[uts_idx])
    epsilon_uts = float(strain_raw[uts_idx])
    
    # STEP 4: YIELD POINT DETECTION
    # For elastomers: use proportional limit (where tangent modulus drops)
    # For metals: use 0.2% offset method
    
    if is_elastomer:
        # ELASTOMER: Use proportional limit method
        # Find where tangent modulus drops to 50% of initial
        window = max(2, len(strain_raw) // 50)
        tangent_moduli = []
        for i in range(window, len(strain_raw) - window):
            local_slope = (stress_smooth[i + window] - stress_smooth[i - window]) / \
                         (strain_raw[i + window] - strain_raw[i - window] + 1e-10)
            tangent_moduli.append(local_slope)
        
        # Find where modulus drops to 50% of initial or strain exceeds 5%
        yield_idx = elastic_break_idx
        initial_modulus = tangent_moduli[0] if tangent_moduli else E
        for i, mod in enumerate(tangent_moduli):
            actual_idx = i + window
            if mod < initial_modulus * 0.5 or strain_raw[actual_idx] > 0.05:
                yield_idx = actual_idx
                break
        
        # Ensure yield_idx is reasonable
        if yield_idx <= 1:
            yield_idx = min(len(strain_raw) // 10, 5)
        
        epsilon_yield = float(strain_raw[yield_idx])
        sigma_yield = float(stress_smooth[yield_idx])
        
        print(f"[DEBUG] Elastomer detected: yield at strain = {epsilon_yield:.4f}, stress = {sigma_yield:.2f} MPa")
        
    else:
        # METAL: 0.2% Offset Method
        offset_strain = 0.002  # 0.2%
        
        # The offset line has the same slope as E but passes through (0.002, 0)
        # Equation: stress = E * (strain - 0.002)
        offset_line_stress = E * (strain_raw - offset_strain)
        
        # Find intersection: where curve stress equals offset line stress
        yield_idx = None
        
        # Find the first index where strain > offset_strain
        search_start = np.searchsorted(strain_raw, offset_strain)
        search_start = max(search_start, 1)
        
        # Look for where curve crosses from above to below the offset line
        diff = stress_smooth - offset_line_stress
        
        print(f"[DEBUG] E = {E:.1f} MPa, offset_strain = {offset_strain}")
        print(f"[DEBUG] search_start = {search_start}, strain at search_start = {strain_raw[search_start]:.4f}")
        
        for i in range(search_start, len(diff) - 1):
            if diff[i] > 0 and diff[i + 1] <= 0:
                yield_idx = i
                print(f"[DEBUG] Found intersection at index {yield_idx}, strain = {strain_raw[yield_idx]:.4f}")
                break
        
        # Fallback methods
        if yield_idx is None:
            print(f"[DEBUG] No intersection found, using fallback")
            if np.all(diff[search_start:uts_idx] > 0):
                yield_idx = elastic_break_idx
            else:
                window = max(2, len(strain_raw) // 100)
                for i in range(elastic_break_idx, min(uts_idx, len(strain_raw) - window)):
                    if strain_raw[i + window] - strain_raw[i] > 0:
                        local_slope = (stress_smooth[i + window] - stress_smooth[i]) / \
                                     (strain_raw[i + window] - strain_raw[i])
                        if local_slope < E * 0.3:
                            yield_idx = i
                            break
        
        if yield_idx is None:
            yield_idx = elastic_break_idx
        
        epsilon_yield = float(strain_raw[yield_idx])
        sigma_yield = float(stress_smooth[yield_idx])
    
    # STEP 5: BREAKING STRESS
    breaking_stress = float(stress_smooth[-1])
    
    # STEP 6: % ELONGATION
    percent_elongation = float(strain_raw[-1]) * 100
    
    # STEP 7: NECKING DETECTION
    has_necking = uts_idx < len(strain_raw) - 5
    necking_start_idx = uts_idx
    
    # STEP 8: RESILIENCE (area under curve to yield)
    resilience = float(np.trapz(stress_smooth[:yield_idx+1], strain_raw[:yield_idx+1]))
    
    # STEP 9: TOUGHNESS (total area)
    toughness = float(np.trapz(stress_smooth, strain_raw))
    
    # STEP 10: STRAIN HARDENING
    n, K = 0, 0
    if uts_idx > yield_idx + 5:
        strain_plastic = strain_raw[yield_idx:uts_idx]
        stress_plastic = stress_smooth[yield_idx:uts_idx]
        
        valid_mask = (strain_plastic > 0) & (stress_plastic > 0)
        if np.sum(valid_mask) > 5:
            try:
                log_strain = np.log(strain_plastic[valid_mask])
                log_stress = np.log(stress_plastic[valid_mask])
                coeffs = np.polyfit(log_strain, log_stress, 1)
                n = float(coeffs[0])
                K = float(np.exp(coeffs[1]))
            except:
                pass
    
    # STEP 10: GENERATE CLEAN GRAPH
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set style
    ax.set_facecolor('#fafafa')
    fig.patch.set_facecolor('white')
    
    # Plot raw data points (subtle)
    ax.scatter(strain_raw, stress_raw, c='#cccccc', s=8, alpha=0.5, label='Raw Data', zorder=1)
    
    # Plot smoothed curve with region colors
    if is_elastomer:
        # For elastomers: single smooth curve (no distinct elastic/plastic regions)
        ax.plot(strain_raw, stress_smooth, color='#8b5cf6', linewidth=2.5, 
                label=f'Stress-Strain (E₀ = {E:.1f} MPa)', zorder=3)
        
        # Mark proportional limit instead of yield
        ax.plot(epsilon_yield, sigma_yield, 'o', color='#f59e0b', markersize=12,
                markeredgecolor='#d97706', markeredgewidth=2, zorder=5)
        ax.annotate(f'Prop. Limit\n{sigma_yield:.2f} MPa', 
                    xy=(epsilon_yield, sigma_yield),
                    xytext=(epsilon_yield + strain_raw[-1]*0.1, sigma_yield * 1.3),
                    fontsize=10, fontweight='bold', color='#d97706',
                    arrowprops=dict(arrowstyle='->', color='#d97706', lw=1.5),
                    ha='left')
        
        # Mark maximum stress
        ax.plot(epsilon_uts, sigma_uts, 'o', color='#ec4899', markersize=12,
                markeredgecolor='#be185d', markeredgewidth=2, zorder=5)
        ax.annotate(f'Max Stress\n{sigma_uts:.2f} MPa', 
                    xy=(epsilon_uts, sigma_uts),
                    xytext=(epsilon_uts - strain_raw[-1]*0.1, sigma_uts * 0.8),
                    fontsize=10, fontweight='bold', color='#be185d',
                    arrowprops=dict(arrowstyle='->', color='#be185d', lw=1.5),
                    ha='right')
    else:
        # For metals: distinct regions
        # Elastic region (blue) - from start to elastic break
        ax.plot(strain_raw[:elastic_break_idx+1], stress_smooth[:elastic_break_idx+1], 
                color='#2563eb', linewidth=2.5, label=f'Elastic (E = {E/1000:.1f} GPa)', zorder=3)
        
        # Transition region (elastic break to yield) - still blue-ish but lighter
        if yield_idx > elastic_break_idx:
            ax.plot(strain_raw[elastic_break_idx:yield_idx+1], stress_smooth[elastic_break_idx:yield_idx+1], 
                    color='#60a5fa', linewidth=2.5, zorder=3)
        
        # Plastic region (green) - from yield to UTS
        if uts_idx > yield_idx:
            ax.plot(strain_raw[yield_idx:uts_idx+1], stress_smooth[yield_idx:uts_idx+1], 
                    color='#16a34a', linewidth=2.5, label='Plastic', zorder=3)
        
        # Necking region (red) - after UTS
        if has_necking and necking_start_idx < len(strain_raw) - 1:
            ax.plot(strain_raw[necking_start_idx:], stress_smooth[necking_start_idx:], 
                    color='#dc2626', linewidth=2.5, label='Necking', zorder=3)
        
        # 0.2% offset line (dashed) - draw from (0.002, 0) to intersection point
        offset_x = np.linspace(0.002, epsilon_yield + 0.005, 50)
        offset_y = E * (offset_x - 0.002)
        # Only plot where stress is positive and below yield stress
        valid = (offset_y >= 0) & (offset_y <= sigma_yield * 1.05)
        if np.any(valid):
            ax.plot(offset_x[valid], offset_y[valid], '--', color='#0891b2', 
                    linewidth=2, alpha=0.9, label='0.2% Offset Line', zorder=4)
        
        # Mark key points
        # Yield point
        ax.plot(epsilon_yield, sigma_yield, 'o', color='#f59e0b', markersize=12,
                markeredgecolor='#d97706', markeredgewidth=2, zorder=5)
        ax.annotate(f'Yield\n{sigma_yield:.0f} MPa', 
                    xy=(epsilon_yield, sigma_yield),
                    xytext=(epsilon_yield + strain_raw[-1]*0.08, sigma_yield * 0.85),
                    fontsize=10, fontweight='bold', color='#d97706',
                    arrowprops=dict(arrowstyle='->', color='#d97706', lw=1.5),
                    ha='left')
        
        # UTS point
        ax.plot(epsilon_uts, sigma_uts, 'o', color='#ec4899', markersize=12,
                markeredgecolor='#be185d', markeredgewidth=2, zorder=5)
        ax.annotate(f'UTS\n{sigma_uts:.0f} MPa', 
                    xy=(epsilon_uts, sigma_uts),
                    xytext=(epsilon_uts - strain_raw[-1]*0.08, sigma_uts * 1.05),
                    fontsize=10, fontweight='bold', color='#be185d',
                    arrowprops=dict(arrowstyle='->', color='#be185d', lw=1.5),
                    ha='right')
    
    # Light grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='#e5e7eb')
    ax.set_axisbelow(True)
    
    # Labels
    ax.set_xlabel('Strain (ε)', fontsize=12, fontweight='bold', color='#374151')
    ax.set_ylabel('Stress (σ) [MPa]', fontsize=12, fontweight='bold', color='#374151')
    
    # Title based on material type
    if is_elastomer:
        ax.set_title('Stress-Strain Curve (Elastomer)', fontsize=14, fontweight='bold', color='#1f2937', pad=15)
    else:
        ax.set_title('Stress-Strain Curve', fontsize=14, fontweight='bold', color='#1f2937', pad=15)
    
    # Clean legend
    ax.legend(loc='lower right', fontsize=9, framealpha=0.95, edgecolor='#e5e7eb')
    
    # Set axis limits with padding
    ax.set_xlim(-strain_raw[-1]*0.02, strain_raw[-1]*1.05)
    ax.set_ylim(-sigma_uts*0.02, sigma_uts*1.15)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#9ca3af')
    ax.spines['bottom'].set_color('#9ca3af')
    
    plt.tight_layout()
    
    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    graph_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    # Downsample arrays for interactive chart (max 500 points)
    max_points = 500
    n_points = len(strain_raw)
    if n_points > max_points:
        indices = np.linspace(0, n_points - 1, max_points, dtype=int)
        # Ensure key indices are included
        key_indices = [0, elastic_break_idx, yield_idx, uts_idx, n_points - 1]
        indices = np.unique(np.concatenate([indices, [i for i in key_indices if 0 <= i < n_points]]))
        indices = np.sort(indices)
    else:
        indices = np.arange(n_points)
    
    # Prepare chart data arrays (downsampled)
    chart_strain = strain_raw[indices].tolist()
    chart_stress_raw = stress_raw[indices].tolist()
    chart_stress_smooth = stress_smooth[indices].tolist()
    
    # Map key indices to downsampled positions
    def find_downsampled_idx(original_idx, indices_arr):
        pos = np.searchsorted(indices_arr, original_idx)
        if pos >= len(indices_arr):
            return len(indices_arr) - 1
        return int(pos)
    
    return {
        'method': 'Mathematical (Elastomer)' if is_elastomer else 'Mathematical',
        'materialType': 'Elastomer' if is_elastomer else 'Metal/Polymer',
        'youngsModulus': round(float(E), decimal_places),
        'yieldStress': round(float(sigma_yield), decimal_places),
        'yieldStrain': round(float(epsilon_yield), decimal_places + 3),
        'UTS': round(float(sigma_uts), decimal_places),
        'percentElongation': round(float(percent_elongation), decimal_places),
        'resilience': round(float(resilience), decimal_places),
        'toughness': round(float(toughness), decimal_places),
        'breakingStress': round(float(breaking_stress), decimal_places),
        'strainHardeningExponent': round(float(n), decimal_places + 1) if n > 0 else 0,
        'strengthCoefficient': round(float(K), decimal_places) if K > 0 else 0,
        'elasticDataPoints': int(elastic_break_idx),
        'plasticDataPoints': int(uts_idx - elastic_break_idx) if uts_idx > elastic_break_idx else 0,
        'neckingDataPoints': int(len(strain_raw) - uts_idx),
        'graphData': f'data:image/png;base64,{graph_base64}',
        # Interactive chart data (downsampled)
        'chartData': {
            'strain': chart_strain,
            'stressRaw': chart_stress_raw,
            'stressSmooth': chart_stress_smooth,
            'yieldIndex': find_downsampled_idx(yield_idx, indices),
            'utsIndex': find_downsampled_idx(uts_idx, indices),
            'elasticEndIndex': find_downsampled_idx(elastic_break_idx, indices),
            'yieldPoint': {'strain': float(epsilon_yield), 'stress': float(sigma_yield)},
            'utsPoint': {'strain': float(epsilon_uts), 'stress': float(sigma_uts)},
            'offsetLine': {
                'x': [0.002, float(epsilon_yield + 0.005)],
                'y': [0, float(E * (epsilon_yield + 0.003))]
            }
        }
    }


# ============================================================================
# MACHINE LEARNING APPROACH
# ============================================================================
def analyze_ml(df, decimal_places):
    """Machine Learning approach using Linear Regression
    
    Special handling for elastomers (elongation > 100%):
    - Uses polynomial fitting instead of linear elastic model
    - Uses Mooney-Rivlin inspired approach for better rubber behavior
    """
    
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline
    
    strain_raw = df['Strain'].values
    stress_raw = df['Stress'].values
    
    # DETECT ELASTOMER: High elongation (>50%) and low stress (<100 MPa max)
    max_strain = float(strain_raw.max())
    max_stress = float(stress_raw.max())
    is_elastomer = max_strain > 0.5 and max_stress < 100  # >50% strain, <100 MPa
    
    if is_elastomer:
        # =====================================================================
        # ELASTOMER MODEL: Polynomial fitting (hyperelastic-inspired)
        # =====================================================================
        
        # For elastomers, use polynomial regression (degree 3 for S-curve)
        poly_degree = 3
        poly_model = make_pipeline(PolynomialFeatures(poly_degree), LinearRegression())
        X = strain_raw.reshape(-1, 1)
        poly_model.fit(X, stress_raw)
        
        # Get coefficients for display
        poly_coeffs = poly_model.named_steps['linearregression'].coef_
        
        # Calculate initial tangent modulus (derivative at origin)
        # For polynomial: stress = a0 + a1*strain + a2*strain^2 + a3*strain^3
        # E_initial = a1 (derivative at strain=0)
        E = float(poly_coeffs[1]) if len(poly_coeffs) > 1 else float(stress_raw[1] / strain_raw[1])
        
        # For elastomers, "yield" is not well-defined, use proportional limit
        # Find where tangent modulus drops to 50% of initial
        strain_fit = np.linspace(0.001, max_strain, 500)
        stress_fit = poly_model.predict(strain_fit.reshape(-1, 1))
        
        # Calculate tangent modulus along curve
        d_stress = np.diff(stress_fit)
        d_strain = np.diff(strain_fit)
        tangent_modulus = d_stress / d_strain
        
        # Find "yield" where modulus drops significantly or use 2% strain
        yield_idx = 0
        for i in range(len(tangent_modulus)):
            if tangent_modulus[i] < E * 0.5 or strain_fit[i] > 0.02:
                yield_idx = i
                break
        
        if yield_idx == 0:
            yield_idx = min(10, len(strain_fit) - 1)
        
        epsilon_yield = float(strain_fit[yield_idx])
        sigma_yield = float(stress_fit[yield_idx])
        
        # UTS is the maximum stress
        uts_idx = np.argmax(stress_raw)
        sigma_uts = float(stress_raw[uts_idx])
        epsilon_uts = float(strain_raw[uts_idx])
        
        # K and n don't apply well to elastomers, use apparent values
        K = sigma_uts / (max_strain ** 0.5) if max_strain > 0 else 1
        n = 0.5  # Typical for rubber-like materials
        
        # Elongation
        percent_elongation = max_strain * 100
        
        # Resilience (area under curve to yield)
        resilience = float(np.trapz(stress_fit[:yield_idx+1], strain_fit[:yield_idx+1]))
        
        # Toughness (total area under curve)
        toughness = float(np.trapz(stress_raw, strain_raw))
        
        # R² for polynomial fit
        stress_pred = poly_model.predict(X)
        ss_res = np.sum((stress_raw - stress_pred) ** 2)
        ss_tot = np.sum((stress_raw - np.mean(stress_raw)) ** 2)
        elastic_r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # GENERATE ELASTOMER GRAPH
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_facecolor('#fafafa')
        fig.patch.set_facecolor('white')
        
        # Scatter raw data
        ax.scatter(strain_raw, stress_raw, c='#cccccc', s=8, alpha=0.5, label='Raw Data', zorder=1)
        
        # Plot polynomial fit
        ax.plot(strain_fit, stress_fit, color='#8b5cf6', linewidth=2.5,
               label=f'Polynomial Fit (degree {poly_degree})', zorder=3)
        
        # Mark key points
        ax.plot(epsilon_yield, sigma_yield, 'o', color='#f59e0b', markersize=10,
               markeredgecolor='#d97706', markeredgewidth=2, zorder=5)
        ax.annotate(f'Prop. Limit\n{sigma_yield:.2f} MPa', 
                   xy=(epsilon_yield, sigma_yield),
                   xytext=(epsilon_yield + max_strain*0.1, sigma_yield * 1.3),
                   fontsize=10, fontweight='bold', color='#d97706',
                   arrowprops=dict(arrowstyle='->', color='#d97706', lw=1.5))
        
        ax.plot(epsilon_uts, sigma_uts, 'o', color='#ec4899', markersize=10,
               markeredgecolor='#be185d', markeredgewidth=2, zorder=5)
        ax.annotate(f'Max Stress\n{sigma_uts:.2f} MPa', 
                   xy=(epsilon_uts, sigma_uts),
                   xytext=(epsilon_uts - max_strain*0.15, sigma_uts * 0.85),
                   fontsize=10, fontweight='bold', color='#be185d',
                   arrowprops=dict(arrowstyle='->', color='#be185d', lw=1.5))
        
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_xlabel('Strain (ε)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Stress (σ) [MPa]', fontsize=12, fontweight='bold')
        ax.set_title('Stress-Strain Curve - Elastomer Model (Polynomial)', fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='lower right', fontsize=9)
        ax.set_xlim(-max_strain*0.02, max_strain*1.08)
        ax.set_ylim(-max_stress*0.05, max_stress*1.15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        graph_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return {
            'method': 'Machine Learning (Elastomer Model)',
            'youngsModulus': round(float(E), decimal_places),
            'yieldStress': round(float(sigma_yield), decimal_places),
            'yieldStrain': round(float(epsilon_yield), decimal_places + 3),
            'UTS': round(float(sigma_uts), decimal_places),
            'utsStrain': round(float(epsilon_uts), decimal_places + 3),
            'percentElongation': round(float(percent_elongation), decimal_places),
            'resilience': round(float(resilience), decimal_places),
            'toughness': round(float(toughness), decimal_places),
            'strainHardeningExponent': round(float(n), decimal_places + 1),
            'strengthCoefficient': round(float(K), decimal_places),
            'elasticDataPoints': yield_idx,
            'plasticDataPoints': len(strain_raw) - yield_idx,
            'neckingDataPoints': 0,
            'elasticR2': round(float(elastic_r2), 4),
            'materialType': 'Elastomer',
            'graphData': f'data:image/png;base64,{graph_base64}',
            'chartData': {
                'strain': strain_fit.tolist(),
                'stressRaw': stress_raw.tolist()[:500],
                'stressSmooth': stress_fit.tolist(),
                'yieldIndex': yield_idx,
                'utsIndex': int(np.argmax(stress_fit)),
                'yieldPoint': {'strain': epsilon_yield, 'stress': sigma_yield},
                'utsPoint': {'strain': epsilon_uts, 'stress': sigma_uts}
            }
        }
    
    # =====================================================================
    # STANDARD METAL MODEL: Linear Elastic + Hollomon Power Law
    # =====================================================================
    
    # Check if deformation_type exists, otherwise auto-detect
    if 'deformation_type' not in df.columns:
        # Auto-detect: elastic if strain < 0.002
        df['deformation_type'] = df['Strain'].apply(lambda x: 'Elastic' if x < 0.002 else 'Plastic')
    
    elastic_data = df[df['deformation_type'].str.lower() == 'elastic'].copy()
    plastic_data = df[df['deformation_type'].str.lower() == 'plastic'].copy()
    
    if len(elastic_data) < 5:
        # Fallback: use first 20% as elastic
        split_idx = max(5, len(df) // 5)
        elastic_data = df.iloc[:split_idx].copy()
        plastic_data = df.iloc[split_idx:].copy()
    
    if len(plastic_data) < 5:
        plastic_data = df.iloc[len(elastic_data):].copy()
    
    # ELASTIC REGIME: Young's Modulus
    X_elastic = elastic_data[['Strain']].values
    y_elastic = elastic_data['Stress'].values
    
    LR_elastic = LinearRegression()
    LR_elastic.fit(X_elastic, y_elastic)
    E = float(LR_elastic.coef_[0])
    
    # Calculate R² for elastic region
    y_pred_elastic = LR_elastic.predict(X_elastic)
    ss_res = np.sum((y_elastic - y_pred_elastic) ** 2)
    ss_tot = np.sum((y_elastic - np.mean(y_elastic)) ** 2)
    elastic_r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # PLASTIC REGIME: K and n (Hollomon equation)
    strain_plastic = plastic_data['Strain'].values
    stress_plastic = plastic_data['Stress'].values
    
    # Filter valid values for log
    valid_mask = (strain_plastic > 0) & (stress_plastic > 0)
    if np.sum(valid_mask) > 5:
        X_plastic = np.log(strain_plastic[valid_mask]).reshape(-1, 1)
        y_plastic = np.log(stress_plastic[valid_mask])
        
        LR_plastic = LinearRegression()
        LR_plastic.fit(X_plastic, y_plastic)
        
        K = float(np.exp(LR_plastic.intercept_))
        n = float(LR_plastic.coef_[0])
    else:
        K, n = 1000, 0.2  # Default values
    
    # DERIVED PROPERTIES
    # Yield point: E*ε = K*ε^n
    try:
        epsilon_yield = float((E / K) ** (1 / (n - 1)))
        sigma_yield = float(E * epsilon_yield)
    except:
        epsilon_yield = 0.002
        sigma_yield = E * epsilon_yield
    
    # UTS
    epsilon_max = float(strain_raw.max())
    sigma_uts = float(K * (epsilon_max ** n))
    
    # % Elongation
    percent_elongation = epsilon_max * 100
    
    # Resilience
    resilience = float(0.5 * E * epsilon_yield**2)
    
    # Toughness
    try:
        plastic_area = float((K / (n + 1)) * (epsilon_max**(n + 1) - epsilon_yield**(n + 1)))
        toughness = resilience + plastic_area
    except:
        toughness = resilience * 2
    
    # GENERATE CLEAN GRAPH
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set style
    ax.set_facecolor('#fafafa')
    fig.patch.set_facecolor('white')
    
    # Scatter raw data (subtle)
    ax.scatter(strain_raw, stress_raw, c='#cccccc', s=8, alpha=0.5, label='Raw Data', zorder=1)
    
    # Generate smooth fitted curves
    strain_elastic_fit = np.linspace(0, epsilon_yield, 100)
    stress_elastic_fit = E * strain_elastic_fit
    
    strain_plastic_fit = np.linspace(epsilon_yield, epsilon_max, 100)
    stress_plastic_fit = K * strain_plastic_fit**n
    
    # Plot ML models
    ax.plot(strain_elastic_fit, stress_elastic_fit, color='#2563eb', linewidth=2.5,
           label=f'Elastic: σ = Eε (E = {E/1000:.1f} GPa)', zorder=3)
    ax.plot(strain_plastic_fit, stress_plastic_fit, color='#16a34a', linewidth=2.5,
           label=f'Plastic: σ = Kεⁿ (K={K:.0f}, n={n:.2f})', zorder=3)
    
    # Mark yield point
    ax.plot(epsilon_yield, sigma_yield, 'o', color='#f59e0b', markersize=10,
           markeredgecolor='#d97706', markeredgewidth=2, zorder=5)
    ax.annotate(f'Yield\n{sigma_yield:.0f} MPa', 
               xy=(epsilon_yield, sigma_yield),
               xytext=(epsilon_yield + epsilon_max*0.08, sigma_yield * 0.7),
               fontsize=10, fontweight='bold', color='#d97706',
               arrowprops=dict(arrowstyle='->', color='#d97706', lw=1.5),
               ha='left')
    
    # Mark UTS
    ax.plot(epsilon_max, sigma_uts, 'o', color='#ec4899', markersize=10,
           markeredgecolor='#be185d', markeredgewidth=2, zorder=5)
    ax.annotate(f'UTS\n{sigma_uts:.0f} MPa', 
               xy=(epsilon_max, sigma_uts),
               xytext=(epsilon_max - epsilon_max*0.12, sigma_uts * 1.08),
               fontsize=10, fontweight='bold', color='#be185d',
               arrowprops=dict(arrowstyle='->', color='#be185d', lw=1.5),
               ha='right')
    
    # Light grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='#e5e7eb')
    ax.set_axisbelow(True)
    
    # Labels
    ax.set_xlabel('Strain (ε)', fontsize=12, fontweight='bold', color='#374151')
    ax.set_ylabel('Stress (σ) [MPa]', fontsize=12, fontweight='bold', color='#374151')
    ax.set_title('Stress-Strain Curve (ML Model)', fontsize=14, fontweight='bold', color='#1f2937', pad=15)
    
    # Clean legend
    ax.legend(loc='lower right', fontsize=9, framealpha=0.95, edgecolor='#e5e7eb')
    
    # Set axis limits with padding
    ax.set_xlim(-epsilon_max*0.02, epsilon_max*1.08)
    ax.set_ylim(-sigma_uts*0.02, sigma_uts*1.15)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#9ca3af')
    ax.spines['bottom'].set_color('#9ca3af')
    
    plt.tight_layout()
    
    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    graph_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return {
        'method': 'Machine Learning',
        'youngsModulus': round(float(E), decimal_places),
        'yieldStress': round(float(sigma_yield), decimal_places),
        'yieldStrain': round(float(epsilon_yield), decimal_places + 3),
        'UTS': round(float(sigma_uts), decimal_places),
        'percentElongation': round(float(percent_elongation), decimal_places),
        'resilience': round(float(resilience), decimal_places),
        'toughness': round(float(toughness), decimal_places),
        'strainHardeningExponent': round(float(n), decimal_places + 1),
        'strengthCoefficient': round(float(K), decimal_places),
        'elasticDataPoints': len(elastic_data),
        'plasticDataPoints': len(plastic_data),
        'neckingDataPoints': 0,
        'elasticR2': round(float(elastic_r2), 4),
        'graphData': f'data:image/png;base64,{graph_base64}',
        # Interactive chart data for ML
        'chartData': {
            'strain': strain_raw.tolist()[:500],
            'stressRaw': stress_raw.tolist()[:500],
            'stressSmooth': stress_raw.tolist()[:500],  # ML uses raw for display
            'elasticFit': {'x': strain_elastic_fit.tolist(), 'y': stress_elastic_fit.tolist()},
            'plasticFit': {'x': strain_plastic_fit.tolist(), 'y': stress_plastic_fit.tolist()},
            'yieldPoint': {'strain': float(epsilon_yield), 'stress': float(sigma_yield)},
            'utsPoint': {'strain': float(epsilon_max), 'stress': float(sigma_uts)}
        }
    }


# ============================================================================
# MAIN ENDPOINT
# ============================================================================
@app.route('/api/analyze', methods=['POST'])
def analyze_stress_strain():
    """Main analysis endpoint - supports both Mathematical and ML approaches"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get parameters
        decimal_places = int(request.form.get('outputDecimalPlaces', 3))
        smoothing_window = int(request.form.get('smoothingWindow', 11))
        smoothing_order = int(request.form.get('smoothingOrder', 3))
        analysis_method = request.form.get('analysisMethod', 'mathematical')  # 'mathematical' or 'ml'
        
        # Read CSV
        df = pd.read_csv(file)
        
        # Handle different CSV formats
        if 'Strain' not in df.columns or 'Stress' not in df.columns:
            if len(df.columns) >= 2:
                df.columns = ['Strain', 'Stress'] + list(df.columns[2:])
            else:
                return jsonify({'error': 'CSV must have at least 2 columns (Strain, Stress)'}), 400
        
        # Clean and sort data
        df = df.dropna(subset=['Strain', 'Stress']).sort_values('Strain').reset_index(drop=True)
        
        if len(df) < 20:
            return jsonify({'error': 'Need at least 20 data points for accurate analysis'}), 400
        
        # Choose analysis method
        if analysis_method.lower() == 'ml':
            results = analyze_ml(df, decimal_places)
        else:
            results = analyze_mathematical(df, decimal_places, smoothing_window, smoothing_order)
        
        return jsonify(results)
    
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


# ============================================================================
# COMPARISON CHARTS ENDPOINT
# ============================================================================
@app.route('/api/generate-comparison', methods=['POST'])
def generate_comparison_charts():
    """Generate comparison charts for multiple materials"""
    try:
        data = request.get_json()
        materials = data.get('materials', [])
        
        if len(materials) < 2:
            return jsonify({'error': 'Need at least 2 materials for comparison'}), 400
        
        # Extract material names and properties
        names = [m.get('name', f'Material {i+1}') for i, m in enumerate(materials)]
        
        # Properties to compare
        properties = {
            'youngsModulus': ("Young's Modulus", 'GPa', 1000),  # Convert MPa to GPa
            'yieldStress': ('Yield Strength', 'MPa', 1),
            'UTS': ('Ultimate Tensile Strength', 'MPa', 1),
            'percentElongation': ('% Elongation', '%', 1),
            'resilience': ('Resilience', 'MPa', 1),
            'toughness': ('Toughness', 'MPa', 1),
            'strainHardeningExponent': ('Strain Hardening (n)', '', 1)
        }
        
        # Colors for materials
        colors = ['#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#8b5cf6', 
                  '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1']
        
        charts = {}
        
        for prop_key, (prop_name, unit, divisor) in properties.items():
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_facecolor('#fafafa')
            fig.patch.set_facecolor('white')
            
            values = []
            for m in materials:
                val = m.get(prop_key, 0)
                if val is None:
                    val = 0
                values.append(float(val) / divisor)
            
            bars = ax.bar(names, values, color=colors[:len(names)], edgecolor='white', linewidth=2)
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.annotate(f'{val:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 5),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=11, fontweight='bold', color='#374151')
            
            ax.set_ylabel(f'{prop_name} ({unit})' if unit else prop_name, fontsize=12, fontweight='bold', color='#374151')
            ax.set_title(f'{prop_name} Comparison', fontsize=14, fontweight='bold', color='#1f2937', pad=15)
            ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5, color='#e5e7eb')
            ax.set_axisbelow(True)
            
            # Style
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#9ca3af')
            ax.spines['bottom'].set_color('#9ca3af')
            
            # Rotate x labels if many materials
            if len(names) > 4:
                plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            
            # Convert to base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
            buf.seek(0)
            chart_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            charts[prop_key] = f'data:image/png;base64,{chart_base64}'
        
        # Generate combined stress-strain overlay chart
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_facecolor('#fafafa')
        fig.patch.set_facecolor('white')
        
        for i, m in enumerate(materials):
            name = names[i]
            color = colors[i % len(colors)]
            
            # Plot yield and UTS points connected
            yield_strain = m.get('yieldStrain', 0)
            yield_stress = m.get('yieldStress', 0)
            uts = m.get('UTS', 0)
            elongation = m.get('percentElongation', 0) / 100
            
            # Simple representation: origin -> yield -> UTS -> final
            strain_pts = [0, yield_strain, elongation * 0.8, elongation]
            stress_pts = [0, yield_stress, uts, uts * 0.9]
            
            ax.plot(strain_pts, stress_pts, '-o', color=color, linewidth=2.5, 
                   markersize=6, label=name)
        
        ax.set_xlabel('Strain (ε)', fontsize=12, fontweight='bold', color='#374151')
        ax.set_ylabel('Stress (σ) [MPa]', fontsize=12, fontweight='bold', color='#374151')
        ax.set_title('Stress-Strain Comparison (Simplified)', fontsize=14, fontweight='bold', color='#1f2937', pad=15)
        ax.legend(loc='lower right', fontsize=10, framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='#e5e7eb')
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        charts['stressStrainOverlay'] = f'data:image/png;base64,{base64.b64encode(buf.read()).decode("utf-8")}'
        plt.close()
        
        return jsonify({'charts': charts})
    
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


# ============================================================================
# PDF REPORT GENERATION
# ============================================================================

def generate_stress_strain_chart(material_data, width=6, height=4):
    """Generate a stress-strain chart for a single material and return as bytes"""
    fig, ax = plt.subplots(figsize=(width, height))
    
    # Get chart data - backend sends 'stressSmooth' not 'stress'
    chart_data = material_data.get('chartData', {})
    strain = chart_data.get('strain', [])
    stress = chart_data.get('stressSmooth', chart_data.get('stress', []))
    
    if not strain or not stress:
        # No data available
        ax.text(0.5, 0.5, 'No chart data available', ha='center', va='center', transform=ax.transAxes)
    else:
        # Plot the stress-strain curve
        ax.plot(strain, stress, 'b-', linewidth=2, label='Stress-Strain Curve')
        
        # Mark yield point
        yield_strain = material_data.get('yieldStrain', 0)
        yield_stress = material_data.get('yieldStress', 0)
        if yield_strain and yield_stress:
            ax.scatter([yield_strain], [yield_stress], color='green', s=100, zorder=5, label=f'Yield ({yield_stress:.1f} MPa)')
            ax.axhline(y=yield_stress, color='green', linestyle='--', alpha=0.3)
        
        # Mark UTS
        uts = material_data.get('UTS', 0)
        uts_strain = material_data.get('utsStrain', material_data.get('UTSStrain', 0))
        if uts and uts_strain:
            ax.scatter([uts_strain], [uts], color='red', s=100, zorder=5, marker='^', label=f'UTS ({uts:.1f} MPa)')
            ax.axhline(y=uts, color='red', linestyle='--', alpha=0.3)
        
        # Draw 0.2% offset line if we have Young's modulus
        E = material_data.get('youngsModulus', 0)
        if E > 0 and len(strain) > 0:
            max_strain = max(strain) * 0.6
            offset_strain = 0.002
            offset_x = [offset_strain, max_strain]
            offset_y = [0, E * (max_strain - offset_strain)]
            # Only draw if within reasonable range
            if offset_y[1] < max(stress) * 1.5:
                ax.plot(offset_x, offset_y, 'g--', alpha=0.5, linewidth=1, label='0.2% Offset')
    
    ax.set_xlabel('Strain (ε)', fontsize=10)
    ax.set_ylabel('Stress (MPa)', fontsize=10)
    ax.set_title(f"Stress-Strain Curve: {material_data.get('name', 'Material')}", fontsize=12)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    
    # Save to bytes
    img_buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    img_buffer.seek(0)
    
    return img_buffer

def generate_comparison_chart(materials, width=7, height=5):
    """Generate a comparison chart with all materials overlaid"""
    fig, ax = plt.subplots(figsize=(width, height))
    
    colors_list = ['#2E86AB', '#E94F37', '#44AF69', '#F5B700', '#9B59B6', 
                   '#3498DB', '#E74C3C', '#2ECC71', '#F39C12', '#1ABC9C']
    
    has_data = False
    for i, material_data in enumerate(materials):
        chart_data = material_data.get('chartData', {})
        strain = chart_data.get('strain', [])
        stress = chart_data.get('stressSmooth', chart_data.get('stress', []))
        
        if strain and stress:
            color = colors_list[i % len(colors_list)]
            name = material_data.get('name', f'Material {i+1}')[:25]
            ax.plot(strain, stress, color=color, linewidth=2, label=name)
            has_data = True
    
    if not has_data:
        ax.text(0.5, 0.5, 'No chart data available', ha='center', va='center', transform=ax.transAxes)
    
    ax.set_xlabel('Strain (ε)', fontsize=10)
    ax.set_ylabel('Stress (MPa)', fontsize=10)
    ax.set_title('Stress-Strain Curve Comparison', fontsize=12)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    
    # Save to bytes
    img_buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    img_buffer.seek(0)
    
    return img_buffer

@app.route('/api/generate-report', methods=['POST'])
def generate_report():
    """Generate a PDF report with analysis results and graphs"""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        
        data = request.json
        materials = data.get('materials', [])
        include_charts = data.get('includeCharts', True)
        
        if not materials:
            return jsonify({'error': 'No materials provided'}), 400
        
        # Create PDF in memory
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
        
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('Title', parent=styles['Title'], fontSize=20, spaceAfter=20)
        heading_style = ParagraphStyle('Heading', parent=styles['Heading2'], fontSize=14, spaceAfter=10)
        subheading_style = ParagraphStyle('Subheading', parent=styles['Heading3'], fontSize=12, spaceAfter=8)
        normal_style = styles['Normal']
        
        story = []
        
        # Title
        story.append(Paragraph("Stress-Strain Analysis Report", title_style))
        story.append(Paragraph(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
        story.append(Spacer(1, 20))
        
        # Summary Table
        story.append(Paragraph("Material Properties Summary", heading_style))
        
        table_data = [['Material', 'E (GPa)', 'Yield (MPa)', 'UTS (MPa)', 'Elongation (%)', 'Toughness']]
        for m in materials:
            table_data.append([
                m.get('name', 'Unknown')[:20],
                f"{m.get('youngsModulus', 0)/1000:.2f}",
                f"{m.get('yieldStress', 0):.1f}",
                f"{m.get('UTS', 0):.1f}",
                f"{m.get('percentElongation', 0):.2f}",
                f"{m.get('toughness', 0):.1f}"
            ])
        
        table = Table(table_data, colWidths=[1.5*inch, 0.8*inch, 0.9*inch, 0.9*inch, 1*inch, 0.9*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.Color(0.2, 0.4, 0.7)),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.Color(0.95, 0.95, 0.98)),
            ('GRID', (0, 0), (-1, -1), 1, colors.Color(0.8, 0.8, 0.8)),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
        ]))
        story.append(table)
        story.append(Spacer(1, 20))
        
        # Comparison Chart (if multiple materials and charts enabled)
        if include_charts and len(materials) > 1:
            story.append(Paragraph("Comparison Chart", heading_style))
            try:
                comparison_img_buffer = generate_comparison_chart(materials)
                comparison_img = Image(comparison_img_buffer, width=6.5*inch, height=4.5*inch)
                story.append(comparison_img)
            except Exception as e:
                story.append(Paragraph(f"(Chart generation failed: {str(e)})", normal_style))
            story.append(Spacer(1, 20))
        
        # Individual Material Details with Charts
        for i, m in enumerate(materials):
            if i > 0:
                story.append(PageBreak())
            
            story.append(Paragraph(f"{m.get('name', f'Material {i+1}')}", heading_style))
            
            # Material details
            details = [
                f"<b>Young's Modulus:</b> {m.get('youngsModulus', 0)/1000:.3f} GPa ({m.get('youngsModulus', 0):.1f} MPa)",
                f"<b>Yield Stress:</b> {m.get('yieldStress', 0):.2f} MPa at Strain: {m.get('yieldStrain', 0):.5f}",
                f"<b>Ultimate Tensile Strength (UTS):</b> {m.get('UTS', 0):.2f} MPa",
                f"<b>Percent Elongation:</b> {m.get('percentElongation', 0):.2f}%",
                f"<b>Resilience:</b> {m.get('resilience', 0):.3f} MPa",
                f"<b>Toughness:</b> {m.get('toughness', 0):.3f} MPa",
                f"<b>Analysis Method:</b> {m.get('method', 'Mathematical')}"
            ]
            
            for detail in details:
                story.append(Paragraph(detail, normal_style))
            
            story.append(Spacer(1, 15))
            
            # Individual chart
            if include_charts:
                story.append(Paragraph("Stress-Strain Curve", subheading_style))
                try:
                    img_buffer = generate_stress_strain_chart(m)
                    img = Image(img_buffer, width=6*inch, height=4*inch)
                    story.append(img)
                except Exception as e:
                    story.append(Paragraph(f"(Chart generation failed: {str(e)})", normal_style))
                
                story.append(Spacer(1, 10))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        
        pdf_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        return jsonify({
            'pdf': f'data:application/pdf;base64,{pdf_base64}',
            'filename': f'stress_strain_report_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        })
        
    except ImportError:
        return jsonify({'error': 'PDF generation requires reportlab. Install with: pip install reportlab'}), 500
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


# ============================================================================
# EXPORT FORMATS (Excel, JSON, MATLAB)
# ============================================================================
@app.route('/api/export', methods=['POST'])
def export_data():
    """Export analysis results in various formats"""
    try:
        data = request.json
        materials = data.get('materials', [])
        format_type = data.get('format', 'json')  # json, excel, csv
        
        if not materials:
            return jsonify({'error': 'No materials provided'}), 400
        
        if format_type == 'json':
            # Pretty JSON export
            export_data = {
                'exportDate': pd.Timestamp.now().isoformat(),
                'materials': materials
            }
            json_str = pd.io.json.dumps(export_data, indent=2)
            json_base64 = base64.b64encode(json_str.encode()).decode('utf-8')
            return jsonify({
                'data': f'data:application/json;base64,{json_base64}',
                'filename': f'stress_strain_export_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.json'
            })
        
        elif format_type == 'excel':
            # Excel export
            buffer = io.BytesIO()
            
            # Create summary dataframe
            summary_data = []
            for m in materials:
                summary_data.append({
                    'Material': m.get('name', 'Unknown'),
                    'Young\'s Modulus (GPa)': m.get('youngsModulus', 0) / 1000,
                    'Yield Stress (MPa)': m.get('yieldStress', 0),
                    'Yield Strain': m.get('yieldStrain', 0),
                    'UTS (MPa)': m.get('UTS', 0),
                    'Elongation (%)': m.get('percentElongation', 0),
                    'Resilience (MPa)': m.get('resilience', 0),
                    'Toughness (MPa)': m.get('toughness', 0),
                    'Method': m.get('method', 'Mathematical')
                })
            
            df = pd.DataFrame(summary_data)
            
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Summary', index=False)
            
            buffer.seek(0)
            excel_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            
            return jsonify({
                'data': f'data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{excel_base64}',
                'filename': f'stress_strain_export_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
            })
        
        elif format_type == 'csv':
            # CSV export
            summary_data = []
            for m in materials:
                summary_data.append({
                    'Material': m.get('name', 'Unknown'),
                    'Young\'s Modulus (GPa)': m.get('youngsModulus', 0) / 1000,
                    'Yield Stress (MPa)': m.get('yieldStress', 0),
                    'Yield Strain': m.get('yieldStrain', 0),
                    'UTS (MPa)': m.get('UTS', 0),
                    'Elongation (%)': m.get('percentElongation', 0),
                    'Resilience (MPa)': m.get('resilience', 0),
                    'Toughness (MPa)': m.get('toughness', 0),
                    'Method': m.get('method', 'Mathematical')
                })
            
            df = pd.DataFrame(summary_data)
            csv_str = df.to_csv(index=False)
            csv_base64 = base64.b64encode(csv_str.encode()).decode('utf-8')
            
            return jsonify({
                'data': f'data:text/csv;base64,{csv_base64}',
                'filename': f'stress_strain_export_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.csv'
            })
        
        else:
            return jsonify({'error': f'Unsupported format: {format_type}'}), 400
            
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


# ============================================================================
# MATERIAL CLUSTERING / CLASSIFICATION
# ============================================================================
@app.route('/api/cluster-materials', methods=['POST'])
def cluster_materials():
    """Cluster materials based on their mechanical properties"""
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        
        data = request.json
        materials = data.get('materials', [])
        n_clusters = min(data.get('nClusters', 3), len(materials))
        
        if len(materials) < 2:
            return jsonify({'error': 'Need at least 2 materials for clustering'}), 400
        
        # Extract features
        features = []
        names = []
        for m in materials:
            features.append([
                m.get('youngsModulus', 0) / 1000,  # GPa
                m.get('yieldStress', 0),
                m.get('UTS', 0),
                m.get('percentElongation', 0),
                m.get('toughness', 0),
                m.get('resilience', 0)
            ])
            names.append(m.get('name', 'Unknown'))
        
        X = np.array(features)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply K-means clustering
        n_clusters = min(n_clusters, len(materials))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Generate cluster visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: PCA scatter
        ax1 = axes[0]
        ax1.set_facecolor('#fafafa')
        scatter_colors = ['#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#8b5cf6']
        
        for i in range(n_clusters):
            mask = clusters == i
            ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       c=scatter_colors[i % len(scatter_colors)],
                       s=150, alpha=0.7, edgecolors='white', linewidth=2,
                       label=f'Cluster {i+1}')
            
            # Add labels
            for j, (x, y) in enumerate(zip(X_pca[mask, 0], X_pca[mask, 1])):
                name_idx = np.where(mask)[0][j]
                ax1.annotate(names[name_idx], (x, y), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=9, alpha=0.8)
        
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontweight='bold')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontweight='bold')
        ax1.set_title('Material Clusters (PCA)', fontsize=14, fontweight='bold', pad=10)
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Feature comparison by cluster
        ax2 = axes[1]
        feature_names = ['E (GPa)', 'Yield', 'UTS', 'Elong.', 'Tough.', 'Resil.']
        x_pos = np.arange(len(feature_names))
        width = 0.8 / n_clusters
        
        for i in range(n_clusters):
            mask = clusters == i
            cluster_means = X[mask].mean(axis=0)
            ax2.bar(x_pos + i * width, cluster_means, width, 
                   label=f'Cluster {i+1}', color=scatter_colors[i % len(scatter_colors)],
                   alpha=0.8, edgecolor='white')
        
        ax2.set_xticks(x_pos + width * (n_clusters - 1) / 2)
        ax2.set_xticklabels(feature_names)
        ax2.set_ylabel('Feature Value', fontweight='bold')
        ax2.set_title('Average Properties by Cluster', fontsize=14, fontweight='bold', pad=10)
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        cluster_chart = f'data:image/png;base64,{base64.b64encode(buf.read()).decode("utf-8")}'
        plt.close()
        
        # Prepare cluster assignments
        cluster_assignments = []
        for i, (name, cluster) in enumerate(zip(names, clusters)):
            cluster_assignments.append({
                'name': name,
                'cluster': int(cluster) + 1,
                'features': {
                    'youngsModulus': features[i][0],
                    'yieldStress': features[i][1],
                    'UTS': features[i][2],
                    'elongation': features[i][3],
                    'toughness': features[i][4],
                    'resilience': features[i][5]
                }
            })
        
        # Find similar materials within each cluster
        similarity_info = []
        for i in range(n_clusters):
            mask = clusters == i
            cluster_names = [names[j] for j in range(len(names)) if mask[j]]
            similarity_info.append({
                'cluster': i + 1,
                'materials': cluster_names,
                'count': int(np.sum(mask))
            })
        
        return jsonify({
            'clusterChart': cluster_chart,
            'assignments': cluster_assignments,
            'clusters': similarity_info,
            'nClusters': n_clusters,
            'pcaVariance': [float(v) for v in pca.explained_variance_ratio_]
        })
        
    except ImportError:
        return jsonify({'error': 'Clustering requires scikit-learn. Install with: pip install scikit-learn'}), 500
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


# ============================================================================
# SESSION SAVE/LOAD
# ============================================================================
@app.route('/api/save-session', methods=['POST'])
def save_session():
    """Save current session state"""
    try:
        data = request.json
        session_data = {
            'version': '1.0',
            'savedAt': pd.Timestamp.now().isoformat(),
            'parameters': data.get('parameters', {}),
            'materials': data.get('materials', [])
        }
        
        json_str = pd.io.json.dumps(session_data, indent=2)
        json_base64 = base64.b64encode(json_str.encode()).decode('utf-8')
        
        return jsonify({
            'session': f'data:application/json;base64,{json_base64}',
            'filename': f'session_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.json'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/load-session', methods=['POST'])
def load_session():
    """Load a saved session"""
    try:
        if 'file' in request.files:
            file = request.files['file']
            session_data = pd.io.json.loads(file.read().decode('utf-8'))
        else:
            session_data = request.json
        
        if 'version' not in session_data:
            return jsonify({'error': 'Invalid session file format'}), 400
        
        return jsonify({
            'parameters': session_data.get('parameters', {}),
            'materials': session_data.get('materials', []),
            'savedAt': session_data.get('savedAt', 'Unknown')
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# SMART CSV DETECTION
# ============================================================================
@app.route('/api/preview-csv', methods=['POST'])
def preview_csv():
    """Preview and detect CSV format"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        content = file.read().decode('utf-8')
        
        # Detect delimiter
        import csv
        sniffer = csv.Sniffer()
        try:
            dialect = sniffer.sniff(content[:2048])
            delimiter = dialect.delimiter
        except:
            delimiter = ','
        
        # Parse with detected delimiter
        lines = content.strip().split('\n')
        
        # Check if first row is header
        first_row = lines[0].split(delimiter)
        has_header = not all(is_numeric(val.strip()) for val in first_row)
        
        # Get preview data
        df = pd.read_csv(io.StringIO(content), delimiter=delimiter, 
                        header=0 if has_header else None, nrows=10)
        
        # Auto-detect strain/stress columns
        column_mapping = {}
        for col in df.columns:
            col_lower = str(col).lower()
            if 'strain' in col_lower or 'eps' in col_lower or 'ε' in col_lower:
                column_mapping['strain'] = str(col)
            elif 'stress' in col_lower or 'sigma' in col_lower or 'σ' in col_lower:
                column_mapping['stress'] = str(col)
        
        # If not found, assume first two columns
        if 'strain' not in column_mapping and len(df.columns) >= 1:
            column_mapping['strain'] = str(df.columns[0])
        if 'stress' not in column_mapping and len(df.columns) >= 2:
            column_mapping['stress'] = str(df.columns[1])
        
        return jsonify({
            'delimiter': delimiter,
            'hasHeader': has_header,
            'columns': [str(c) for c in df.columns],
            'columnMapping': column_mapping,
            'rowCount': len(lines) - (1 if has_header else 0),
            'preview': df.head(5).to_dict(orient='records'),
            'dataTypes': {str(k): str(v) for k, v in df.dtypes.items()}
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def is_numeric(s):
    """Check if string is numeric"""
    try:
        float(s.replace(',', '.'))
        return True
    except:
        return False


if __name__ == '__main__':
    app.run(debug=True, port=5000)
