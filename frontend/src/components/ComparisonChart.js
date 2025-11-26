import React, { useMemo } from 'react';
import Plot from 'react-plotly.js';

/**
 * Enhanced Interactive Comparison Chart
 * Overlays multiple stress-strain curves with synchronized hover
 */
const ComparisonChart = ({ materials }) => {
  const colors = ['#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#8b5cf6', 
                  '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1'];

  const { data, layout } = useMemo(() => {
    if (!materials || materials.length === 0) {
      return { data: [], layout: {} };
    }

    const traces = [];

    materials.forEach((mat, idx) => {
      const chartData = mat.results?.chartData;
      const color = colors[idx % colors.length];
      
      if (chartData?.strain && chartData?.stressSmooth) {
        // Main curve
        traces.push({
          x: chartData.strain,
          y: chartData.stressSmooth,
          mode: 'lines',
          type: 'scatter',
          name: mat.name,
          line: {
            color: color,
            width: 2.5
          },
          hovertemplate: 
            `<b>${mat.name}</b><br>` +
            'Strain: %{x:.5f}<br>' +
            'Stress: %{y:.2f} MPa<br>' +
            '<extra></extra>'
        });

        // Yield point marker
        if (chartData.yieldPoint) {
          traces.push({
            x: [chartData.yieldPoint.strain],
            y: [chartData.yieldPoint.stress],
            mode: 'markers',
            type: 'scatter',
            name: `${mat.name} Yield`,
            marker: {
              color: color,
              size: 12,
              symbol: 'circle',
              line: { color: 'white', width: 2 }
            },
            showlegend: false,
            hovertemplate: 
              `<b>${mat.name} - Yield</b><br>` +
              `${chartData.yieldPoint.stress.toFixed(1)} MPa<br>` +
              '<extra></extra>'
          });
        }

        // UTS point marker
        if (chartData.utsPoint) {
          traces.push({
            x: [chartData.utsPoint.strain],
            y: [chartData.utsPoint.stress],
            mode: 'markers',
            type: 'scatter',
            name: `${mat.name} UTS`,
            marker: {
              color: color,
              size: 12,
              symbol: 'diamond',
              line: { color: 'white', width: 2 }
            },
            showlegend: false,
            hovertemplate: 
              `<b>${mat.name} - UTS</b><br>` +
              `${chartData.utsPoint.stress.toFixed(1)} MPa<br>` +
              '<extra></extra>'
          });
        }
      } else {
        // Fallback: simplified curve from results
        const results = mat.results;
        if (results) {
          const yieldStrain = results.yieldStrain || 0.002;
          const elongation = (results.percentElongation || 10) / 100;
          
          traces.push({
            x: [0, yieldStrain, elongation * 0.8, elongation],
            y: [0, results.yieldStress || 0, results.UTS || 0, (results.UTS || 0) * 0.9],
            mode: 'lines+markers',
            type: 'scatter',
            name: mat.name,
            line: { color: color, width: 2.5 },
            marker: { size: 8 },
            hovertemplate: 
              `<b>${mat.name}</b><br>` +
              'Strain: %{x:.5f}<br>' +
              'Stress: %{y:.2f} MPa<br>' +
              '<extra></extra>'
          });
        }
      }
    });

    // Calculate axis ranges
    let maxStrain = 0.1, maxStress = 100;
    materials.forEach(mat => {
      const results = mat.results;
      if (results) {
        maxStrain = Math.max(maxStrain, (results.percentElongation || 0) / 100 * 1.1);
        maxStress = Math.max(maxStress, (results.UTS || 0) * 1.15);
      }
    });

    const plotLayout = {
      title: {
        text: 'Stress-Strain Comparison (Interactive)',
        font: { size: 16, color: '#1f2937' }
      },
      xaxis: {
        title: { text: 'Strain (ε)', font: { size: 13 } },
        range: [-maxStrain * 0.02, maxStrain],
        gridcolor: '#e5e7eb',
        zeroline: true
      },
      yaxis: {
        title: { text: 'Stress (σ) [MPa]', font: { size: 13 } },
        range: [-maxStress * 0.02, maxStress],
        gridcolor: '#e5e7eb',
        zeroline: true
      },
      legend: {
        x: 1,
        y: 1,
        xanchor: 'right',
        bgcolor: 'rgba(255,255,255,0.9)',
        bordercolor: '#e5e7eb',
        borderwidth: 1
      },
      hovermode: 'closest',
      hoverlabel: {
        bgcolor: 'white',
        bordercolor: '#667eea'
      },
      plot_bgcolor: '#fafafa',
      paper_bgcolor: 'white',
      margin: { l: 60, r: 30, t: 50, b: 50 }
    };

    return { data: traces, layout: plotLayout };
  }, [materials, colors]);

  if (!materials || materials.length === 0) {
    return (
      <div className="comparison-placeholder">
        <p>Add materials to see comparison chart</p>
      </div>
    );
  }

  return (
    <div className="comparison-chart-container">
      <Plot
        data={data}
        layout={layout}
        config={{
          responsive: true,
          displayModeBar: true,
          displaylogo: false,
          modeBarButtonsToRemove: ['lasso2d', 'select2d'],
          toImageButtonOptions: {
            format: 'png',
            filename: 'stress_strain_comparison',
            height: 600,
            width: 1000,
            scale: 2
          }
        }}
        style={{ width: '100%', height: '500px' }}
      />
      
      <div className="chart-legend-info">
        <span>○ = Yield Point</span>
        <span>◇ = UTS</span>
        <span className="hint">Hover for values • Drag to zoom • Double-click to reset</span>
      </div>
    </div>
  );
};

export default ComparisonChart;
