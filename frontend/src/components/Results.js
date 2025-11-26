import React, { useState } from 'react';
import InteractiveCurve from './InteractiveCurve';

const Results = ({ results, materialName, hideGraph }) => {
  const isMathematical = results.method === 'Mathematical';
  const [useInteractive, setUseInteractive] = useState(true);
  
  // Check if chartData is available for interactive mode
  const hasChartData = results.chartData && results.chartData.strain;
  
  return (
    <div>
      <h2>📊 {materialName ? `${materialName} - ` : ''}Analysis Results</h2>
      
      {/* Method badge */}
      <div style={{
        display: 'inline-block',
        padding: '8px 16px',
        borderRadius: '20px',
        background: isMathematical 
          ? 'linear-gradient(135deg, #3498db 0%, #2980b9 100%)' 
          : 'linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%)',
        color: 'white',
        fontWeight: 'bold',
        fontSize: '14px',
        marginBottom: '15px',
        boxShadow: '0 2px 10px rgba(0,0,0,0.2)'
      }}>
        {isMathematical ? '📐 Mathematical Approach' : '🤖 Machine Learning Approach'}
      </div>
      
      <div className="results-grid">
        <div className="result-item">
          <h3>Young's Modulus (E)</h3>
          <p>{(results.youngsModulus / 1000).toFixed(2)} GPa</p>
        </div>
        
        <div className="result-item">
          <h3>Yield Stress {isMathematical ? '(0.2% Offset)' : '(Model)'}</h3>
          <p>{results.yieldStress} MPa</p>
        </div>
        
        <div className="result-item">
          <h3>Yield Strain</h3>
          <p>{results.yieldStrain}</p>
        </div>
        
        <div className="result-item">
          <h3>UTS</h3>
          <p>{results.UTS} MPa</p>
        </div>
        
        <div className="result-item">
          <h3>% Elongation</h3>
          <p>{results.percentElongation}%</p>
        </div>
        
        <div className="result-item">
          <h3>Resilience</h3>
          <p>{results.resilience} {isMathematical ? 'MPa' : 'MJ/m³'}</p>
        </div>
        
        <div className="result-item">
          <h3>Toughness</h3>
          <p>{results.toughness} {isMathematical ? 'MPa' : 'MJ/m³'}</p>
        </div>
      </div>

      <div style={{ marginTop: '20px', padding: '15px', background: '#f8f9fa', borderRadius: '8px' }}>
        <h3 style={{ marginBottom: '10px', color: '#667eea' }}>
          {isMathematical ? 'Region Distribution & Data Statistics' : 'ML Model Parameters'}
        </h3>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', fontSize: '14px' }}>
          <div>
            <strong>Elastic Data Points:</strong> {results.elasticDataPoints}
          </div>
          <div>
            <strong>Plastic Data Points:</strong> {results.plasticDataPoints}
          </div>
          {isMathematical && (
            <div>
              <strong>Necking Data Points:</strong> {results.neckingDataPoints}
            </div>
          )}
          <div>
            <strong>Elastic R²:</strong> {results.elasticR2}
          </div>
          {(results.strainHardeningExponent > 0 || !isMathematical) && (
            <>
              <div>
                <strong>Strength Coefficient (K):</strong> {results.strengthCoefficient} MPa
              </div>
              <div>
                <strong>Strain Hardening Exp (n):</strong> {results.strainHardeningExponent}
              </div>
            </>
          )}
        </div>
      </div>

      {/* Graph section with toggle */}
      {!hideGraph && (results.graphData || hasChartData) && (
        <div className="graph-container">
          <div style={{ 
            display: 'flex', 
            justifyContent: 'space-between', 
            alignItems: 'center',
            marginBottom: '15px'
          }}>
            <h3 style={{ margin: 0, color: '#667eea' }}>
              Stress-Strain Curve ({results.method} Analysis)
            </h3>
            
            {/* Toggle between interactive and static */}
            {hasChartData && results.graphData && (
              <div className="chart-toggle">
                <button 
                  className={`toggle-btn ${useInteractive ? 'active' : ''}`}
                  onClick={() => setUseInteractive(true)}
                >
                  📈 Interactive
                </button>
                <button 
                  className={`toggle-btn ${!useInteractive ? 'active' : ''}`}
                  onClick={() => setUseInteractive(false)}
                >
                  🖼️ Static
                </button>
              </div>
            )}
          </div>
          
          {/* Render interactive or static chart */}
          {useInteractive && hasChartData ? (
            <InteractiveCurve 
              chartData={results.chartData}
              method={results.method}
              materialName={materialName}
            />
          ) : results.graphData ? (
            <img src={results.graphData} alt="Stress-Strain Curve" />
          ) : null}
        </div>
      )}
    </div>
  );
};

export default Results;
