import React, { useState, useCallback } from 'react';
import axios from 'axios';

/**
 * Smart CSV Importer with format detection and preview
 */
const SmartCSVImporter = ({ onImport, onCancel }) => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [columnMapping, setColumnMapping] = useState({ strain: '', stress: '' });

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.name.endsWith('.csv')) {
      handleFileSelect(droppedFile);
    } else {
      setError('Please drop a CSV file');
    }
  }, []);

  const handleFileSelect = async (selectedFile) => {
    setFile(selectedFile);
    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const response = await axios.post('/api/preview-csv', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      setPreview(response.data);
      setColumnMapping(response.data.columnMapping || { strain: '', stress: '' });

    } catch (err) {
      setError(err.response?.data?.error || 'Failed to preview file');
    } finally {
      setLoading(false);
    }
  };

  const handleImport = () => {
    if (!file || !columnMapping.strain || !columnMapping.stress) {
      setError('Please select strain and stress columns');
      return;
    }

    onImport({
      file,
      columnMapping,
      delimiter: preview?.delimiter || ','
    });
  };

  return (
    <div className="smart-importer">
      <div className="importer-header">
        <h3>📁 Smart CSV Import</h3>
        {onCancel && (
          <button className="close-btn" onClick={onCancel}>✕</button>
        )}
      </div>

      {/* Drop Zone */}
      {!file && (
        <div 
          className="drop-zone"
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
          onDragEnter={(e) => e.currentTarget.classList.add('drag-over')}
          onDragLeave={(e) => e.currentTarget.classList.remove('drag-over')}
        >
          <div className="drop-icon">📄</div>
          <p>Drag & drop your CSV file here</p>
          <span>or</span>
          <label className="file-select-btn">
            Browse Files
            <input 
              type="file" 
              accept=".csv"
              onChange={(e) => e.target.files[0] && handleFileSelect(e.target.files[0])}
            />
          </label>
        </div>
      )}

      {error && (
        <div className="importer-error">{error}</div>
      )}

      {loading && (
        <div className="importer-loading">
          <div className="spinner"></div>
          Analyzing file format...
        </div>
      )}

      {/* Preview Section */}
      {preview && (
        <div className="preview-section">
          <div className="file-info">
            <span className="file-name">📄 {file.name}</span>
            <span className="file-stats">
              {preview.rowCount} rows • Delimiter: "{preview.delimiter}" • 
              {preview.hasHeader ? ' Has header' : ' No header'}
            </span>
          </div>

          {/* Column Mapping */}
          <div className="column-mapping">
            <h4>📋 Map Columns</h4>
            <div className="mapping-row">
              <label>
                Strain Column:
                <select 
                  value={columnMapping.strain}
                  onChange={(e) => setColumnMapping({...columnMapping, strain: e.target.value})}
                >
                  <option value="">Select...</option>
                  {preview.columns.map((col, idx) => (
                    <option key={idx} value={col}>{col}</option>
                  ))}
                </select>
              </label>

              <label>
                Stress Column:
                <select 
                  value={columnMapping.stress}
                  onChange={(e) => setColumnMapping({...columnMapping, stress: e.target.value})}
                >
                  <option value="">Select...</option>
                  {preview.columns.map((col, idx) => (
                    <option key={idx} value={col}>{col}</option>
                  ))}
                </select>
              </label>
            </div>
          </div>

          {/* Data Preview Table */}
          <div className="data-preview">
            <h4>👀 Data Preview (first 5 rows)</h4>
            <div className="preview-table-container">
              <table className="preview-table">
                <thead>
                  <tr>
                    {preview.columns.map((col, idx) => (
                      <th 
                        key={idx}
                        className={
                          col === columnMapping.strain ? 'strain-col' :
                          col === columnMapping.stress ? 'stress-col' : ''
                        }
                      >
                        {col}
                        {col === columnMapping.strain && <span className="col-badge">Strain</span>}
                        {col === columnMapping.stress && <span className="col-badge">Stress</span>}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {preview.preview.map((row, idx) => (
                    <tr key={idx}>
                      {preview.columns.map((col, colIdx) => (
                        <td 
                          key={colIdx}
                          className={
                            col === columnMapping.strain ? 'strain-col' :
                            col === columnMapping.stress ? 'stress-col' : ''
                          }
                        >
                          {row[col]}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Actions */}
          <div className="importer-actions">
            <button className="btn secondary" onClick={() => {
              setFile(null);
              setPreview(null);
            }}>
              ← Choose Different File
            </button>
            <button 
              className="btn primary"
              onClick={handleImport}
              disabled={!columnMapping.strain || !columnMapping.stress}
            >
              ✓ Import & Analyze
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default SmartCSVImporter;
