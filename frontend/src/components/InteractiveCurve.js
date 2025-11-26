import React, { useMemo, useState } from 'react';
import Plot from 'react-plotly.js';

/**
 * Interactive Stress-Strain Curve using Plotly
 * - Hover snaps to nearest point on the curve
 * - Shows yield point and UTS markers
 * - Displays data point info on hover
 * - Optional uncertainty bands
 */
const InteractiveCurve = ({ chartData, method, materialName, showUncertainty = true }) => {
  const [hoverInfo, setHoverInfo] = useState(null);
  
  const isMathematical = method === 'Mathematical';
  
  // Calculate uncertainty bands (residual-based)
  const calculateUncertaintyBands = (strain, stressRaw, stressSmooth) => {
    if (!strain || !stressRaw || !stressSmooth || strain.length < 10) return null;
    
    // Calculate residuals
    const residuals = stressRaw.map((raw, i) => Math.abs(raw - stressSmooth[i]));
    
    // Rolling std dev (window of ~5% of data)
    const windowSize = Math.max(5, Math.floor(strain.length * 0.05));
    const upperBand = [];
    const lowerBand = [];
    
    for (let i = 0; i < strain.length; i++) {
      const start = Math.max(0, i - Math.floor(windowSize / 2));
      const end = Math.min(strain.length, i + Math.floor(windowSize / 2));
      const windowResiduals = residuals.slice(start, end);
      const mean = windowResiduals.reduce((a, b) => a + b, 0) / windowResiduals.length;
      const variance = windowResiduals.reduce((a, b) => a + (b - mean) ** 2, 0) / windowResiduals.length;
      const std = Math.sqrt(variance);
      
      upperBand.push(stressSmooth[i] + 1.5 * std);
      lowerBand.push(Math.max(0, stressSmooth[i] - 1.5 * std));
    }
    
    return { upperBand, lowerBand };
  };
  
  // Build Plotly traces
  const { data, layout } = useMemo(() => {
    if (!chartData || !chartData.strain) {
      return { data: [], layout: {} };
    }
    
    const traces = [];
    const { strain, stressRaw, stressSmooth, yieldPoint, utsPoint } = chartData;
    
    if (isMathematical) {
      // Mathematical approach: show raw scatter + smooth curve
      
      // Uncertainty bands (if enabled)
      if (showUncertainty && stressRaw && stressSmooth) {
        const bands = calculateUncertaintyBands(strain, stressRaw, stressSmooth);
        if (bands) {
          // Upper band
          traces.push({
            x: strain,
            y: bands.upperBand,
            mode: 'lines',
            type: 'scatter',
            name: 'Upper Bound',
            line: { color: 'transparent' },
            showlegend: false,
            hoverinfo: 'skip'
          });
          
          // Lower band with fill
          traces.push({
            x: strain,
            y: bands.lowerBand,
            mode: 'lines',
            type: 'scatter',
            name: 'Uncertainty (±1.5σ)',
            fill: 'tonexty',
            fillcolor: 'rgba(37, 99, 235, 0.15)',
            line: { color: 'transparent' },
            hoverinfo: 'skip'
          });
        }
      }
      
      // Raw data points (subtle scatter)
      traces.push({
        x: strain,
        y: stressRaw,
        mode: 'markers',
        type: 'scatter',
        name: 'Raw Data',
        marker: {
          color: '#cccccc',
          size: 5,
          opacity: 0.5
        },
        hoverinfo: 'skip'
      });
      
      // Smoothed curve (main interactive line)
      traces.push({
        x: strain,
        y: stressSmooth,
        mode: 'lines',
        type: 'scatter',
        name: 'Stress-Strain Curve',
        line: {
          color: '#2563eb',
          width: 2.5
        },
        hovertemplate: 
          '<b>Strain:</b> %{x:.5f}<br>' +
          '<b>Stress:</b> %{y:.2f} MPa<br>' +
          '<extra></extra>'
      });
      
      // 0.2% Offset line if available
      if (chartData.offsetLine) {
        traces.push({
          x: chartData.offsetLine.x,
          y: chartData.offsetLine.y,
          mode: 'lines',
          type: 'scatter',
          name: '0.2% Offset Line',
          line: {
            color: '#0891b2',
            width: 2,
            dash: 'dash'
          },
          hoverinfo: 'skip'
        });
      }
      
    } else {
      // ML approach: show raw scatter + fitted curves
      
      // Raw data points
      traces.push({
        x: strain,
        y: stressRaw,
        mode: 'markers',
        type: 'scatter',
        name: 'Raw Data',
        marker: {
          color: '#cccccc',
          size: 5,
          opacity: 0.5
        },
        hovertemplate: 
          '<b>Strain:</b> %{x:.5f}<br>' +
          '<b>Stress:</b> %{y:.2f} MPa<br>' +
          '<extra></extra>'
      });
      
      // Elastic fit line
      if (chartData.elasticFit) {
        traces.push({
          x: chartData.elasticFit.x,
          y: chartData.elasticFit.y,
          mode: 'lines',
          type: 'scatter',
          name: 'Elastic Fit (σ = Eε)',
          line: {
            color: '#2563eb',
            width: 2.5
          },
          hovertemplate: 
            '<b>Elastic Region</b><br>' +
            'Strain: %{x:.5f}<br>' +
            'Stress: %{y:.2f} MPa<br>' +
            '<extra></extra>'
        });
      }
      
      // Plastic fit line
      if (chartData.plasticFit) {
        traces.push({
          x: chartData.plasticFit.x,
          y: chartData.plasticFit.y,
          mode: 'lines',
          type: 'scatter',
          name: 'Plastic Fit (σ = Kεⁿ)',
          line: {
            color: '#16a34a',
            width: 2.5
          },
          hovertemplate: 
            '<b>Plastic Region</b><br>' +
            'Strain: %{x:.5f}<br>' +
            'Stress: %{y:.2f} MPa<br>' +
            '<extra></extra>'
        });
      }
    }
    
    // Yield point marker
    if (yieldPoint) {
      traces.push({
        x: [yieldPoint.strain],
        y: [yieldPoint.stress],
        mode: 'markers+text',
        type: 'scatter',
        name: 'Yield Point',
        marker: {
          color: '#f59e0b',
          size: 14,
          symbol: 'circle',
          line: {
            color: '#d97706',
            width: 2
          }
        },
        text: [`Yield: ${yieldPoint.stress.toFixed(0)} MPa`],
        textposition: 'top right',
        textfont: {
          color: '#d97706',
          size: 11,
          family: 'Arial, sans-serif'
        },
        hovertemplate: 
          '<b>YIELD POINT</b><br>' +
          'Strain: %{x:.5f}<br>' +
          'Stress: %{y:.2f} MPa<br>' +
          '<extra></extra>'
      });
    }
    
    // UTS point marker
    if (utsPoint) {
      traces.push({
        x: [utsPoint.strain],
        y: [utsPoint.stress],
        mode: 'markers+text',
        type: 'scatter',
        name: 'UTS',
        marker: {
          color: '#ec4899',
          size: 14,
          symbol: 'circle',
          line: {
            color: '#be185d',
            width: 2
          }
        },
        text: [`UTS: ${utsPoint.stress.toFixed(0)} MPa`],
        textposition: 'top left',
        textfont: {
          color: '#be185d',
          size: 11,
          family: 'Arial, sans-serif'
        },
        hovertemplate: 
          '<b>ULTIMATE TENSILE STRENGTH</b><br>' +
          'Strain: %{x:.5f}<br>' +
          'Stress: %{y:.2f} MPa<br>' +
          '<extra></extra>'
      });
    }
    
    // Calculate axis ranges
    const maxStrain = Math.max(...strain) * 1.08;
    const maxStress = Math.max(...(stressSmooth || stressRaw)) * 1.15;
    
    const plotLayout = {
      title: {
        text: `Stress-Strain Curve${materialName ? ` - ${materialName}` : ''} (${method})`,
        font: {
          size: 16,
          color: '#1f2937',
          family: 'Arial, sans-serif'
        }
      },
      xaxis: {
        title: {
          text: 'Strain (ε)',
          font: { size: 13, color: '#374151' }
        },
        range: [-maxStrain * 0.02, maxStrain],
        gridcolor: '#e5e7eb',
        gridwidth: 1,
        zeroline: true,
        zerolinecolor: '#9ca3af',
        zerolinewidth: 1
      },
      yaxis: {
        title: {
          text: 'Stress (σ) [MPa]',
          font: { size: 13, color: '#374151' }
        },
        range: [-maxStress * 0.02, maxStress],
        gridcolor: '#e5e7eb',
        gridwidth: 1,
        zeroline: true,
        zerolinecolor: '#9ca3af',
        zerolinewidth: 1
      },
      legend: {
        x: 1,
        y: 0,
        xanchor: 'right',
        yanchor: 'bottom',
        bgcolor: 'rgba(255,255,255,0.9)',
        bordercolor: '#e5e7eb',
        borderwidth: 1,
        font: { size: 10 }
      },
      hovermode: 'closest',
      hoverlabel: {
        bgcolor: 'white',
        bordercolor: '#667eea',
        font: { size: 12, color: '#1f2937' }
      },
      plot_bgcolor: '#fafafa',
      paper_bgcolor: 'white',
      margin: { l: 60, r: 30, t: 50, b: 50 },
      showlegend: true
    };
    
    return { data: traces, layout: plotLayout };
  }, [chartData, method, materialName, isMathematical, showUncertainty]);
  
  // Handle hover events
  const handleHover = (event) => {
    if (event.points && event.points.length > 0) {
      const point = event.points[0];
      setHoverInfo({
        strain: point.x,
        stress: point.y,
        curveName: point.data.name
      });
    }
  };
  
  const handleUnhover = () => {
    setHoverInfo(null);
  };
  
  if (!chartData || !chartData.strain) {
    return (
      <div className="interactive-curve-placeholder">
        <p>No chart data available</p>
      </div>
    );
  }
  
  return (
    <div className="interactive-curve-container">
      <Plot
        data={data}
        layout={layout}
        config={{
          responsive: true,
          displayModeBar: true,
          modeBarButtonsToRemove: ['lasso2d', 'select2d'],
          displaylogo: false,
          toImageButtonOptions: {
            format: 'png',
            filename: `stress_strain_${materialName || 'curve'}`,
            height: 600,
            width: 900,
            scale: 2
          }
        }}
        style={{ width: '100%', height: '450px' }}
        onHover={handleHover}
        onUnhover={handleUnhover}
      />
      
      {/* Hover info panel */}
      {hoverInfo && (
        <div className="hover-info-panel">
          <div className="hover-info-item">
            <span className="hover-label">Curve:</span>
            <span className="hover-value">{hoverInfo.curveName}</span>
          </div>
          <div className="hover-info-item">
            <span className="hover-label">Strain (ε):</span>
            <span className="hover-value">{hoverInfo.strain.toFixed(5)}</span>
          </div>
          <div className="hover-info-item">
            <span className="hover-label">Stress (σ):</span>
            <span className="hover-value">{hoverInfo.stress.toFixed(2)} MPa</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default InteractiveCurve;
