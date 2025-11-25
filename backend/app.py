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
    """
    
    strain_raw = df['Strain'].values.astype(float)
    stress_raw = df['Stress'].values.astype(float)
    
    # STEP 1: SMOOTH THE DATA
    if len(strain_raw) < smoothing_window:
        smoothing_window = max(5, len(strain_raw) // 4)
        if smoothing_window % 2 == 0:
            smoothing_window += 1
    
    stress_smooth = savgol_filter(stress_raw, smoothing_window, min(smoothing_order, smoothing_window-1))
    
    # STEP 2: FIND ELASTIC MODULUS - Detect where curve deviates from linearity
    def find_elastic_region(strain, stress):
        """
        Find the elastic region by detecting where the curve starts to deviate 
        from the initial linear portion. Uses derivative analysis.
        """
        n = len(strain)
        
        # Calculate initial slope from first few points (skip very first points which may have noise)
        start_idx = max(2, n // 50)
        initial_end = min(start_idx + 10, n // 4)
        
        # Get initial slope estimate
        initial_slope, initial_intercept, _, _, _ = scipy.stats.linregress(
            strain[start_idx:initial_end], stress[start_idx:initial_end]
        )
        
        # Now find where the actual stress deviates significantly from the linear prediction
        # Calculate the deviation at each point
        predicted_stress = initial_slope * strain + initial_intercept
        deviation = np.abs(stress - predicted_stress) / (np.abs(stress) + 1e-10)
        
        # Find where deviation exceeds threshold (5%)
        elastic_end_idx = initial_end
        deviation_threshold = 0.05
        
        for i in range(initial_end, min(n, n // 2)):
            if deviation[i] > deviation_threshold:
                elastic_end_idx = i
                break
            elastic_end_idx = i
        
        # Refine: also check where slope changes significantly
        # Calculate local slopes using a sliding window
        window = max(3, n // 100)
        for i in range(initial_end, min(elastic_end_idx + 20, n - window)):
            local_slope = (stress[i + window] - stress[i]) / (strain[i + window] - strain[i] + 1e-10)
            if local_slope < initial_slope * 0.7:  # Slope dropped to 70% of initial
                elastic_end_idx = min(elastic_end_idx, i)
                break
        
        # Final regression on the identified elastic region
        if elastic_end_idx > start_idx + 5:
            E, intercept, _, _, _ = scipy.stats.linregress(
                strain[start_idx:elastic_end_idx], stress[start_idx:elastic_end_idx]
            )
        else:
            E, intercept = initial_slope, initial_intercept
            elastic_end_idx = initial_end
        
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
    
    # STEP 4: 0.2% OFFSET YIELD POINT
    offset_strain = 0.002  # 0.2%
    
    # The offset line has the same slope as E but passes through (0.002, 0)
    # Equation: stress = E * (strain - 0.002)
    offset_line_stress = E * (strain_raw - offset_strain)
    
    # Find intersection: where curve stress equals offset line stress
    # Start searching from after the offset point
    yield_idx = None
    
    # Find the first index where strain > offset_strain
    search_start = np.searchsorted(strain_raw, offset_strain)
    search_start = max(search_start, 5)
    
    # Look for where curve crosses from above to below the offset line
    diff = stress_smooth - offset_line_stress
    
    for i in range(search_start, len(diff) - 1):
        # Initially curve is above offset line (diff > 0)
        # Yield point is where it crosses (diff changes sign)
        if diff[i] > 0 and diff[i + 1] <= 0:
            yield_idx = i
            break
    
    # Fallback: if no intersection found, use proportional limit method
    if yield_idx is None:
        # Find where slope drops significantly
        window = max(3, len(strain_raw) // 100)
        for i in range(elastic_break_idx + window, min(uts_idx, len(strain_raw) - window)):
            if strain_raw[i + window] - strain_raw[i - window] > 0:
                local_slope = (stress_smooth[i + window] - stress_smooth[i - window]) / \
                             (strain_raw[i + window] - strain_raw[i - window])
                if local_slope < E * 0.3:  # Slope dropped to 30% of E
                    yield_idx = i
                    break
    
    # Final fallback: use a point where stress is ~90% of UTS in early curve
    if yield_idx is None:
        target_stress = sigma_uts * 0.9
        for i in range(len(stress_smooth)):
            if stress_smooth[i] >= target_stress:
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
    
    return {
        'method': 'Mathematical',
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
        'graphData': f'data:image/png;base64,{graph_base64}'
    }


# ============================================================================
# MACHINE LEARNING APPROACH
# ============================================================================
def analyze_ml(df, decimal_places):
    """Machine Learning approach using Linear Regression"""
    
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    
    strain_raw = df['Strain'].values
    stress_raw = df['Stress'].values
    
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
        'graphData': f'data:image/png;base64,{graph_base64}'
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


if __name__ == '__main__':
    app.run(debug=True, port=5000)
