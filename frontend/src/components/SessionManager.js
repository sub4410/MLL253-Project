import React, { useState, useRef } from 'react';
import axios from 'axios';

/**
 * Session Manager - Save and Load analysis sessions
 */
const SessionManager = ({ materials, parameters, onLoadSession }) => {
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState(null);
  const fileInputRef = useRef(null);

  const handleSaveSession = async () => {
    if (!materials || materials.length === 0) {
      setMessage({ type: 'error', text: 'No materials to save' });
      return;
    }

    setLoading(true);
    setMessage(null);

    try {
      const materialsData = materials.map(m => ({
        name: m.name,
        fileName: m.fileName,
        results: m.results
      }));

      const response = await axios.post('/api/save-session', {
        parameters,
        materials: materialsData
      });

      // Download the session file
      const link = document.createElement('a');
      link.href = response.data.session;
      link.download = response.data.filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

      setMessage({ type: 'success', text: 'Session saved successfully!' });

    } catch (err) {
      setMessage({ type: 'error', text: err.response?.data?.error || 'Failed to save session' });
    } finally {
      setLoading(false);
    }
  };

  const handleLoadClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setLoading(true);
    setMessage(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await axios.post('/api/load-session', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      if (onLoadSession) {
        onLoadSession({
          parameters: response.data.parameters,
          materials: response.data.materials,
          savedAt: response.data.savedAt
        });
      }

      setMessage({ 
        type: 'success', 
        text: `Session loaded! (${response.data.materials?.length || 0} materials)` 
      });

    } catch (err) {
      setMessage({ type: 'error', text: err.response?.data?.error || 'Failed to load session' });
    } finally {
      setLoading(false);
      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  return (
    <div className="session-manager">
      <div className="session-buttons">
        <button 
          className="session-btn save"
          onClick={handleSaveSession}
          disabled={loading || !materials || materials.length === 0}
          title="Save current session"
        >
          💾 Save Session
        </button>
        
        <button 
          className="session-btn load"
          onClick={handleLoadClick}
          disabled={loading}
          title="Load a saved session"
        >
          📂 Load Session
        </button>
        
        <input
          ref={fileInputRef}
          type="file"
          accept=".json"
          onChange={handleFileChange}
          style={{ display: 'none' }}
        />
      </div>

      {message && (
        <div className={`session-message ${message.type}`}>
          {message.text}
        </div>
      )}

      {loading && (
        <div className="session-loading">
          Processing...
        </div>
      )}
    </div>
  );
};

export default SessionManager;
