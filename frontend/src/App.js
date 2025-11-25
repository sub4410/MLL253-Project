import React, { useState } from 'react';
import './App.css';
import FileUpload from './components/FileUpload';
import ParametersForm from './components/ParametersForm';
import Results from './components/Results';
import ComparisonResults from './components/ComparisonResults';
import MaterialSelector from './components/MaterialSelector';
import axios from 'axios';

function App() {
  const [parameters, setParameters] = useState({
    outputDecimalPlaces: 3,
    smoothingWindow: 11,
    smoothingOrder: 3,
    analysisMethod: 'mathematical',
  });
  
  // Input mode: 'upload' or 'library'
  const [inputMode, setInputMode] = useState('upload');
  
  // Multi-file state
  const [allResults, setAllResults] = useState([]); // Array of {name, results}
  const [pendingFile, setPendingFile] = useState(null);
  const [comparisonCharts, setComparisonCharts] = useState(null);
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileUpload = (uploadedFile) => {
    setPendingFile(uploadedFile);
    setError(null);
  };

  const handleParametersChange = (newParameters) => {
    setParameters(newParameters);
  };

  // Handle material selection from library
  const handleSelectMaterial = async (materialId, materialName) => {
    if (allResults.length >= 10) {
      setError('Maximum 10 materials allowed. Please reset to add more.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await axios.post('/api/analyze-material', {
        materialId,
        analysisMethod: parameters.analysisMethod,
        outputDecimalPlaces: parameters.outputDecimalPlaces,
        smoothingWindow: parameters.smoothingWindow,
        smoothingOrder: parameters.smoothingOrder
      });

      const newResult = {
        name: materialName,
        fileName: `${materialId} (Library)`,
        results: response.data
      };
      
      const updatedResults = [...allResults, newResult];
      setAllResults(updatedResults);
      
      if (updatedResults.length >= 2) {
        await generateComparison(updatedResults);
      }
      
    } catch (err) {
      setError(err.response?.data?.error || 'An error occurred during analysis');
      console.error('Analysis error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleAnalyze = async () => {
    if (!pendingFile) {
      setError('Please upload a CSV file first');
      return;
    }

    if (allResults.length >= 10) {
      setError('Maximum 10 files allowed. Please reset to add more.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', pendingFile);
      
      Object.keys(parameters).forEach(key => {
        formData.append(key, parameters[key]);
      });

      const response = await axios.post('/api/analyze', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      // Add to results array with file name
      const materialName = pendingFile.name.replace('.csv', '').replace(/_/g, ' ');
      const newResult = {
        name: materialName,
        fileName: pendingFile.name,
        results: response.data
      };
      
      const updatedResults = [...allResults, newResult];
      setAllResults(updatedResults);
      setPendingFile(null);
      
      // Generate comparison charts if more than 1 result
      if (updatedResults.length >= 2) {
        await generateComparison(updatedResults);
      }
      
    } catch (err) {
      setError(err.response?.data?.error || 'An error occurred during analysis');
      console.error('Analysis error:', err);
    } finally {
      setLoading(false);
    }
  };

  const generateComparison = async (results) => {
    try {
      const materials = results.map(r => ({
        name: r.name,
        ...r.results
      }));
      
      const response = await axios.post('/api/generate-comparison', { materials });
      setComparisonCharts(response.data.charts);
    } catch (err) {
      console.error('Comparison generation error:', err);
    }
  };

  const handleReset = () => {
    setAllResults([]);
    setPendingFile(null);
    setComparisonCharts(null);
    setError(null);
  };

  const handleRemoveMaterial = async (index) => {
    const updatedResults = allResults.filter((_, i) => i !== index);
    setAllResults(updatedResults);
    
    if (updatedResults.length >= 2) {
      await generateComparison(updatedResults);
    } else {
      setComparisonCharts(null);
    }
  };

  return (
    <div className="App">
      <header className="app-header">
        <h1>🔬 Stress-Strain Analyzer</h1>
        <p>Analyze and compare mechanical properties of multiple materials</p>
        {allResults.length > 0 && (
          <div className="material-count">
            📁 {allResults.length}/10 materials loaded
          </div>
        )}
      </header>

      {error && (
        <div className="error">
          <strong>Error:</strong> {error}
        </div>
      )}

      {/* Input Mode Tabs */}
      <div className="input-mode-tabs">
        <button 
          className={`mode-tab ${inputMode === 'upload' ? 'active' : ''}`}
          onClick={() => setInputMode('upload')}
        >
          📤 Upload CSV
        </button>
        <button 
          className={`mode-tab ${inputMode === 'library' ? 'active' : ''}`}
          onClick={() => setInputMode('library')}
        >
          📚 Material Library
        </button>
        {allResults.length > 0 && (
          <button 
            className="mode-tab reset-tab" 
            onClick={handleReset}
            disabled={loading}
          >
            🔄 Reset All
          </button>
        )}
      </div>

      <div className="main-container">
        {/* Left Panel - Input Selection */}
        <div className="card input-panel">
          {inputMode === 'upload' ? (
            <>
              <FileUpload onFileUpload={handleFileUpload} currentFile={pendingFile} />
              <ParametersForm 
                parameters={parameters} 
                onParametersChange={handleParametersChange} 
              />
              
              <div className="button-group">
                <button 
                  className="button primary" 
                  onClick={handleAnalyze}
                  disabled={!pendingFile || loading || allResults.length >= 10}
                >
                  {loading ? 'Analyzing...' : allResults.length === 0 ? 'Run Analysis' : 'Add & Analyze'}
                </button>
              </div>
            </>
          ) : (
            <>
              <ParametersForm 
                parameters={parameters} 
                onParametersChange={handleParametersChange}
                compact={true}
              />
              <MaterialSelector 
                onSelectMaterial={handleSelectMaterial}
                disabled={loading}
              />
            </>
          )}
          
          {/* Material List */}
          {allResults.length > 0 && (
            <div className="material-list">
              <h3>📋 Loaded Materials ({allResults.length}/10)</h3>
              {allResults.map((item, index) => (
                <div key={index} className="material-item">
                  <span className="material-color" style={{ 
                    backgroundColor: ['#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#8b5cf6', 
                                      '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1'][index] 
                  }}></span>
                  <span className="material-name">{item.name}</span>
                  <button 
                    className="remove-btn" 
                    onClick={() => handleRemoveMaterial(index)}
                    title="Remove material"
                  >
                    ✕
                  </button>
                </div>
              ))}
              {allResults.length < 10 && (
                <p className="add-more-hint">+ Add more files to compare (up to 10)</p>
              )}
            </div>
          )}
        </div>

        {/* Right Panel - Results */}
        <div className="card results-panel">
          {loading && (
            <div className="loading">
              <div className="loading-spinner"></div>
              <p>Analyzing your data...</p>
            </div>
          )}
          
          {/* Show individual results */}
          {allResults.length > 0 && !loading && (
            <div className="results-container">
              {/* Tabs for multiple results */}
              {allResults.length === 1 ? (
                <Results results={allResults[0].results} materialName={allResults[0].name} />
              ) : (
                <>
                  {/* Comparison Section */}
                  <ComparisonResults 
                    materials={allResults} 
                    charts={comparisonCharts}
                  />
                  
                  {/* Individual Results Accordion */}
                  <div className="individual-results">
                    <h2>📊 Individual Material Results</h2>
                    {allResults.map((item, index) => (
                      <details key={index} className="material-details">
                        <summary style={{
                          borderLeftColor: ['#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#8b5cf6', 
                                           '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1'][index]
                        }}>
                          {item.name}
                        </summary>
                        <div className="details-content">
                          <Results results={item.results} materialName={item.name} hideGraph={true} />
                        </div>
                      </details>
                    ))}
                  </div>
                </>
              )}
            </div>
          )}
          
          {allResults.length === 0 && !loading && (
            <div className="empty-state">
              <h2>📊 Results</h2>
              <p>Upload a CSV file and click "Run Analysis" to see results</p>
              <div className="feature-list">
                <p>✨ Features:</p>
                <ul>
                  <li>Analyze up to 10 materials</li>
                  <li>Compare mechanical properties</li>
                  <li>Visual comparison charts</li>
                  <li>Mathematical or ML analysis</li>
                </ul>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
