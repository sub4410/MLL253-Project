import React, { useState } from 'react';

const ComparisonResults = ({ materials, charts }) => {
  const [activeChart, setActiveChart] = useState('youngsModulus');
  
  const chartOptions = [
    { key: 'youngsModulus', label: "Young's Modulus", icon: '📏' },
    { key: 'yieldStress', label: 'Yield Strength', icon: '💪' },
    { key: 'UTS', label: 'UTS', icon: '🎯' },
    { key: 'percentElongation', label: '% Elongation', icon: '📐' },
    { key: 'resilience', label: 'Resilience', icon: '🔋' },
    { key: 'toughness', label: 'Toughness', icon: '🛡️' },
    { key: 'strainHardeningExponent', label: 'Strain Hardening', icon: '📈' },
  ];

  const colors = ['#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#8b5cf6', 
                  '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1'];

  if (!materials || materials.length < 2) {
    return null;
  }

  return (
    <div className="comparison-results">
      <h2>📊 Material Comparison</h2>
      <p className="comparison-subtitle">
        Comparing {materials.length} materials
      </p>
      
      {/* Comparison Table */}
      <div className="comparison-table-wrapper">
        <table className="comparison-table">
          <thead>
            <tr>
              <th>Property</th>
              {materials.map((m, i) => (
                <th key={i} style={{ borderTopColor: colors[i] }}>
                  <span className="material-dot" style={{ backgroundColor: colors[i] }}></span>
                  {m.name}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Young's Modulus (GPa)</td>
              {materials.map((m, i) => (
                <td key={i}>{(m.results.youngsModulus / 1000).toFixed(1)}</td>
              ))}
            </tr>
            <tr>
              <td>Yield Strength (MPa)</td>
              {materials.map((m, i) => (
                <td key={i}>{m.results.yieldStress?.toFixed(1)}</td>
              ))}
            </tr>
            <tr>
              <td>UTS (MPa)</td>
              {materials.map((m, i) => (
                <td key={i}>{m.results.UTS?.toFixed(1)}</td>
              ))}
            </tr>
            <tr>
              <td>% Elongation</td>
              {materials.map((m, i) => (
                <td key={i}>{m.results.percentElongation?.toFixed(2)}%</td>
              ))}
            </tr>
            <tr>
              <td>Resilience (MPa)</td>
              {materials.map((m, i) => (
                <td key={i}>{m.results.resilience?.toFixed(3)}</td>
              ))}
            </tr>
            <tr>
              <td>Toughness (MPa)</td>
              {materials.map((m, i) => (
                <td key={i}>{m.results.toughness?.toFixed(2)}</td>
              ))}
            </tr>
            <tr>
              <td>Strain Hardening (n)</td>
              {materials.map((m, i) => (
                <td key={i}>{m.results.strainHardeningExponent?.toFixed(4) || 'N/A'}</td>
              ))}
            </tr>
          </tbody>
        </table>
      </div>
      
      {/* Chart Selection */}
      {charts && (
        <div className="charts-section">
          <h3>📈 Comparison Charts</h3>
          
          <div className="chart-tabs">
            {chartOptions.map(opt => (
              <button 
                key={opt.key}
                className={`chart-tab ${activeChart === opt.key ? 'active' : ''}`}
                onClick={() => setActiveChart(opt.key)}
              >
                {opt.icon} {opt.label}
              </button>
            ))}
          </div>
          
          <div className="chart-display">
            {charts[activeChart] && (
              <img 
                src={charts[activeChart]} 
                alt={`${activeChart} comparison`}
                className="comparison-chart"
              />
            )}
          </div>
          
          {/* All Charts Grid */}
          <details className="all-charts-section">
            <summary>📊 View All Charts</summary>
            <div className="charts-grid">
              {chartOptions.map(opt => (
                charts[opt.key] && (
                  <div key={opt.key} className="chart-item">
                    <h4>{opt.icon} {opt.label}</h4>
                    <img src={charts[opt.key]} alt={opt.label} />
                  </div>
                )
              ))}
            </div>
          </details>
        </div>
      )}
      
      {/* Individual Stress-Strain Curves */}
      <div className="individual-curves">
        <h3>📉 Individual Stress-Strain Curves</h3>
        <div className="curves-grid">
          {materials.map((m, i) => (
            m.results.graphData && (
              <div key={i} className="curve-item" style={{ borderColor: colors[i] }}>
                <h4 style={{ color: colors[i] }}>{m.name}</h4>
                <img src={m.results.graphData} alt={`${m.name} curve`} />
              </div>
            )
          ))}
        </div>
      </div>
    </div>
  );
};

export default ComparisonResults;
