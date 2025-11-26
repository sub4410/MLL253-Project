import React, { useState } from 'react';
import axios from 'axios';

/**
 * Export Panel - PDF Reports, Excel, JSON, CSV exports
 */
const ExportPanel = ({ materials, onClose }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleExport = async (format) => {
    if (!materials || materials.length === 0) {
      setError('No materials to export');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const materialsData = materials.map(m => ({
        name: m.name,
        ...m.results,
        // Include chart data for PDF graphs
        chartData: m.results?.chartData || null
      }));

      let response;
      
      if (format === 'pdf') {
        response = await axios.post('/api/generate-report', {
          materials: materialsData,
          includeCharts: true
        });
        downloadFile(response.data.pdf, response.data.filename);
      } else {
        response = await axios.post('/api/export', {
          materials: materialsData,
          format: format
        });
        downloadFile(response.data.data, response.data.filename);
      }

    } catch (err) {
      setError(err.response?.data?.error || 'Export failed');
      console.error('Export error:', err);
    } finally {
      setLoading(false);
    }
  };

  const downloadFile = (dataUrl, filename) => {
    const link = document.createElement('a');
    link.href = dataUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="export-panel">
      <div className="export-header">
        <h3>📤 Export Results</h3>
        {onClose && (
          <button className="close-btn" onClick={onClose}>✕</button>
        )}
      </div>

      {error && (
        <div className="export-error">
          {error}
        </div>
      )}

      <div className="export-options">
        <button 
          className="export-btn pdf"
          onClick={() => handleExport('pdf')}
          disabled={loading}
        >
          <span className="export-icon">📄</span>
          <span className="export-label">PDF Report</span>
          <span className="export-desc">Full analysis report with tables</span>
        </button>

        <button 
          className="export-btn excel"
          onClick={() => handleExport('excel')}
          disabled={loading}
        >
          <span className="export-icon">📊</span>
          <span className="export-label">Excel (.xlsx)</span>
          <span className="export-desc">Spreadsheet with all data</span>
        </button>

        <button 
          className="export-btn json"
          onClick={() => handleExport('json')}
          disabled={loading}
        >
          <span className="export-icon">{ }</span>
          <span className="export-label">JSON</span>
          <span className="export-desc">Machine-readable format</span>
        </button>

        <button 
          className="export-btn csv"
          onClick={() => handleExport('csv')}
          disabled={loading}
        >
          <span className="export-icon">📋</span>
          <span className="export-label">CSV</span>
          <span className="export-desc">Simple comma-separated</span>
        </button>
      </div>

      {loading && (
        <div className="export-loading">
          <div className="spinner"></div>
          Generating export...
        </div>
      )}

      <div className="export-info">
        <p>📁 {materials?.length || 0} material(s) will be exported</p>
      </div>
    </div>
  );
};

export default ExportPanel;
