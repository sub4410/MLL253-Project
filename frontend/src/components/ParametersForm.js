import React from 'react';

const ParametersForm = ({ parameters, onParametersChange, compact }) => {
  const handleChange = (e) => {
    const { name, value } = e.target;
    const newParameters = {
      ...parameters,
      [name]: name === 'analysisMethod' ? value : (parseFloat(value) || value),
    };
    onParametersChange(newParameters);
  };

  const isMathematical = parameters.analysisMethod === 'mathematical';

  return (
    <div style={{ marginTop: compact ? '0' : '20px' }}>
      {!compact && <h2>⚙️ Analysis Parameters</h2>}
      
      {/* Analysis Method Selection */}
      <div style={{ 
        marginBottom: compact ? '15px' : '20px', 
        padding: compact ? '12px' : '15px', 
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', 
        borderRadius: '10px',
        boxShadow: '0 4px 15px rgba(102, 126, 234, 0.3)'
      }}>
        <label style={{ 
          display: 'block', 
          marginBottom: '10px', 
          color: 'white', 
          fontWeight: 'bold',
          fontSize: '16px'
        }}>
          🎯 Choose Analysis Method
        </label>
        <div style={{ display: 'flex', gap: '10px' }}>
          <button
            type="button"
            onClick={() => onParametersChange({ ...parameters, analysisMethod: 'mathematical' })}
            style={{
              flex: 1,
              padding: '12px 15px',
              border: 'none',
              borderRadius: '8px',
              cursor: 'pointer',
              fontWeight: 'bold',
              fontSize: '14px',
              transition: 'all 0.3s ease',
              background: isMathematical ? 'white' : 'rgba(255,255,255,0.2)',
              color: isMathematical ? '#667eea' : 'white',
              boxShadow: isMathematical ? '0 4px 10px rgba(0,0,0,0.2)' : 'none'
            }}
          >
            📐 Mathematical
          </button>
          <button
            type="button"
            onClick={() => onParametersChange({ ...parameters, analysisMethod: 'ml' })}
            style={{
              flex: 1,
              padding: '12px 15px',
              border: 'none',
              borderRadius: '8px',
              cursor: 'pointer',
              fontWeight: 'bold',
              fontSize: '14px',
              transition: 'all 0.3s ease',
              background: !isMathematical ? 'white' : 'rgba(255,255,255,0.2)',
              color: !isMathematical ? '#764ba2' : 'white',
              boxShadow: !isMathematical ? '0 4px 10px rgba(0,0,0,0.2)' : 'none'
            }}
          >
            🤖 Machine Learning
          </button>
        </div>
      </div>

      <div className="form-group">
        <label htmlFor="outputDecimalPlaces">Decimal Places</label>
        <input
          type="number"
          id="outputDecimalPlaces"
          name="outputDecimalPlaces"
          value={parameters.outputDecimalPlaces}
          onChange={handleChange}
          min="0"
          max="10"
        />
      </div>

      {/* Show smoothing options only for Mathematical approach */}
      {isMathematical && (
        <>
          <div className="form-group">
            <label htmlFor="smoothingWindow">Smoothing Window Size</label>
            <input
              type="number"
              id="smoothingWindow"
              name="smoothingWindow"
              value={parameters.smoothingWindow || 11}
              onChange={handleChange}
              min="5"
              max="51"
              step="2"
              title="Window size for Savitzky-Golay smoothing filter (must be odd)"
            />
          </div>

          <div className="form-group">
            <label htmlFor="smoothingOrder">Smoothing Polynomial Order</label>
            <input
              type="number"
              id="smoothingOrder"
              name="smoothingOrder"
              value={parameters.smoothingOrder || 3}
              onChange={handleChange}
              min="2"
              max="5"
              title="Polynomial order for smoothing (2-5)"
            />
          </div>
        </>
      )}

      {/* Method description */}
      <div style={{ 
        marginTop: '15px', 
        padding: '15px', 
        background: isMathematical ? '#e8f4fd' : '#f3e8fd', 
        borderRadius: '8px',
        borderLeft: `4px solid ${isMathematical ? '#3498db' : '#9b59b6'}`
      }}>
        {isMathematical ? (
          <>
            <p style={{ fontSize: '14px', color: '#2c3e50', margin: 0 }}>
              <strong>📐 Mathematical Approach:</strong>
            </p>
            <ul style={{ fontSize: '13px', color: '#555', margin: '8px 0 0 0', paddingLeft: '20px' }}>
              <li>Savitzky-Golay filter for smooth curves</li>
              <li>R² threshold for elastic region detection</li>
              <li><strong>0.2% offset</strong> yield point (ASTM E8)</li>
              <li>Numerical integration for resilience/toughness</li>
              <li>Region marking: Elastic → Plastic → Necking</li>
            </ul>
          </>
        ) : (
          <>
            <p style={{ fontSize: '14px', color: '#2c3e50', margin: 0 }}>
              <strong>🤖 Machine Learning Approach:</strong>
            </p>
            <ul style={{ fontSize: '13px', color: '#555', margin: '8px 0 0 0', paddingLeft: '20px' }}>
              <li>Linear Regression for elastic region (σ = Eε)</li>
              <li>Log-transformed LR for plastic region (σ = Kεⁿ)</li>
              <li>Yield at model intersection point</li>
              <li>Analytical formulas for energy calculations</li>
              <li>Fitted smooth curves (not raw data)</li>
            </ul>
          </>
        )}
        <p style={{ fontSize: '12px', color: '#666', marginTop: '10px', marginBottom: 0 }}>
          CSV format: <strong>Strain, Stress</strong> (2 columns minimum)
        </p>
      </div>
    </div>
  );
};

export default ParametersForm;
