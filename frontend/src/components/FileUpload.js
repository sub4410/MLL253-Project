import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';

const FileUpload = ({ onFileUpload, currentFile }) => {
  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      onFileUpload(file);
    }
  }, [onFileUpload]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.ms-excel': ['.csv']
    },
    multiple: false
  });

  return (
    <div style={{ marginBottom: '20px' }}>
      <h2>📁 Upload CSV File</h2>
      <div
        {...getRootProps()}
        style={{
          border: '3px dashed #667eea',
          borderRadius: '8px',
          padding: '30px',
          textAlign: 'center',
          cursor: 'pointer',
          background: isDragActive ? '#f0f4ff' : '#fafafa',
          transition: 'all 0.3s',
        }}
      >
        <input {...getInputProps()} />
        {isDragActive ? (
          <p style={{ color: '#667eea', fontSize: '16px', margin: 0 }}>
            📥 Drop the CSV file here...
          </p>
        ) : (
          <div>
            <p style={{ fontSize: '16px', color: '#666', marginBottom: '10px' }}>
              🖱️ Drag & drop a CSV file here, or click to select
            </p>
            <p style={{ fontSize: '12px', color: '#999' }}>
              CSV format: <strong>Strain, Stress</strong> (minimum 2 columns, 20+ rows)
            </p>
            <p style={{ fontSize: '11px', color: '#aaa', marginTop: '5px' }}>
              Works with both Mathematical and Machine Learning analysis
            </p>
          </div>
        )}
      </div>
      
      {currentFile && (
        <div
          style={{
            marginTop: '15px',
            padding: '12px',
            background: '#e8f5e9',
            borderRadius: '6px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}
        >
          <div>
            <strong style={{ color: '#2e7d32' }}>✓ File selected:</strong>
            <span style={{ marginLeft: '10px', color: '#555' }}>
              {currentFile.name}
            </span>
            <span style={{ marginLeft: '10px', color: '#999', fontSize: '12px' }}>
              ({(currentFile.size / 1024).toFixed(2)} KB)
            </span>
          </div>
        </div>
      )}
    </div>
  );
};

export default FileUpload;
