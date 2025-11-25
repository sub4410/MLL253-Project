import React, { useState, useEffect } from 'react';
import axios from 'axios';

const MaterialSelector = ({ onSelectMaterial, disabled }) => {
  const [materials, setMaterials] = useState([]);
  const [categories, setCategories] = useState([]);
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [selectedSource, setSelectedSource] = useState('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showCitation, setShowCitation] = useState(null);

  useEffect(() => {
    fetchMaterials();
  }, []);

  const fetchMaterials = async () => {
    try {
      setLoading(true);
      const response = await axios.get('/api/materials');
      setMaterials(response.data.materials);
      setCategories(response.data.categories);
      setError(null);
    } catch (err) {
      setError('Failed to load material library');
      console.error('Error fetching materials:', err);
    } finally {
      setLoading(false);
    }
  };

  const filteredMaterials = materials.filter(mat => {
    const matchesCategory = selectedCategory === 'all' || mat.category === selectedCategory;
    const matchesSource = selectedSource === 'all' || mat.data_source === selectedSource;
    const matchesSearch = mat.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         mat.description.toLowerCase().includes(searchTerm.toLowerCase());
    return matchesCategory && matchesSource && matchesSearch;
  });

  // Count materials by source
  const realCount = materials.filter(m => m.data_source === 'real').length;
  const refCount = materials.filter(m => m.data_source === 'reference').length;

  const categoryIcons = {
    'Steel': '🔩',
    'Aluminum': '✈️',
    'Titanium': '🚀',
    'Copper': '🔌',
    'Nickel Alloy': '🔥',
    'Polymer': '🧪',
    'Magnesium': '🪶',
    'Rubber/Elastomer': '🧲',
    'Natural Fiber': '🌿'
  };

  const categoryColors = {
    'Steel': '#3b82f6',
    'Aluminum': '#8b5cf6',
    'Titanium': '#06b6d4',
    'Copper': '#f59e0b',
    'Nickel Alloy': '#ef4444',
    'Polymer': '#22c55e',
    'Magnesium': '#ec4899',
    'Rubber/Elastomer': '#14b8a6',
    'Natural Fiber': '#84cc16'
  };

  if (loading) {
    return (
      <div className="material-selector loading-state">
        <div className="loading-spinner small"></div>
        <p>Loading material library...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="material-selector error-state">
        <p>❌ {error}</p>
        <button onClick={fetchMaterials} className="retry-btn">Retry</button>
      </div>
    );
  }

  return (
    <div className="material-selector">
      <h2>📚 Material Library</h2>
      <p className="selector-subtitle">
        Select from {materials.length} materials: 
        <span className="source-badge real">🔬 {realCount} Experimental</span>
        <span className="source-badge ref">📖 {refCount} Reference</span>
      </p>
      
      {/* Search and Filter */}
      <div className="selector-controls">
        <input
          type="text"
          placeholder="🔍 Search materials..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="search-input"
        />
        
        {/* Data Source Filter */}
        <div className="source-filters">
          <button 
            className={`source-btn ${selectedSource === 'all' ? 'active' : ''}`}
            onClick={() => setSelectedSource('all')}
          >
            All Data
          </button>
          <button 
            className={`source-btn real ${selectedSource === 'real' ? 'active' : ''}`}
            onClick={() => setSelectedSource('real')}
          >
            🔬 Experimental
          </button>
          <button 
            className={`source-btn ref ${selectedSource === 'reference' ? 'active' : ''}`}
            onClick={() => setSelectedSource('reference')}
          >
            📖 Reference
          </button>
        </div>
        
        <div className="category-filters">
          <button 
            className={`category-btn ${selectedCategory === 'all' ? 'active' : ''}`}
            onClick={() => setSelectedCategory('all')}
          >
            All
          </button>
          {categories.map(cat => (
            <button 
              key={cat}
              className={`category-btn ${selectedCategory === cat ? 'active' : ''}`}
              onClick={() => setSelectedCategory(cat)}
              style={{ 
                borderColor: selectedCategory === cat ? categoryColors[cat] : undefined,
                backgroundColor: selectedCategory === cat ? categoryColors[cat] : undefined
              }}
            >
              {categoryIcons[cat] || '📦'} {cat}
            </button>
          ))}
        </div>
      </div>

      {/* Material Grid */}
      <div className="material-grid">
        {filteredMaterials.length === 0 ? (
          <div className="no-results">
            <p>No materials found matching your criteria</p>
          </div>
        ) : (
          filteredMaterials.map(mat => (
            <div 
              key={mat.id} 
              className={`material-card ${mat.data_source === 'real' ? 'experimental' : 'reference'}`}
              style={{ borderTopColor: categoryColors[mat.category] || '#666' }}
            >
              <div className="material-header">
                <span className="material-icon">{categoryIcons[mat.category] || '📦'}</span>
                <span className="material-category" style={{ color: categoryColors[mat.category] || '#666' }}>
                  {mat.category}
                </span>
                <span className={`data-source-tag ${mat.data_source}`}>
                  {mat.data_source === 'real' ? '🔬 Experimental' : '📖 Reference'}
                </span>
              </div>
              
              <h3 className="material-title">{mat.name}</h3>
              <p className="material-desc">{mat.description}</p>
              
              {/* Citation Info */}
              {mat.citation && mat.data_source === 'real' && (
                <div className="citation-section">
                  <button 
                    className="citation-toggle"
                    onClick={() => setShowCitation(showCitation === mat.id ? null : mat.id)}
                  >
                    📄 {showCitation === mat.id ? 'Hide' : 'Show'} Citation
                  </button>
                  {showCitation === mat.id && (
                    <div className="citation-details">
                      <p><strong>Authors:</strong> {mat.citation.authors}</p>
                      <p><strong>Source:</strong> {mat.citation.journal} ({mat.citation.year})</p>
                      <p><strong>Institution:</strong> {mat.citation.institution}</p>
                      {mat.citation.doi && (
                        <p>
                          <strong>DOI:</strong>{' '}
                          <a href={`https://doi.org/${mat.citation.doi}`} target="_blank" rel="noopener noreferrer">
                            {mat.citation.doi}
                          </a>
                        </p>
                      )}
                    </div>
                  )}
                </div>
              )}
              
              <div className="material-props">
                <div className="prop">
                  <span className="prop-label">E</span>
                  <span className="prop-value">{mat.properties.E > 1000 ? (mat.properties.E / 1000).toFixed(0) + ' GPa' : mat.properties.E.toFixed(0) + ' MPa'}</span>
                </div>
                <div className="prop">
                  <span className="prop-label">σy</span>
                  <span className="prop-value">{mat.properties.yield_stress.toFixed(0)} MPa</span>
                </div>
                <div className="prop">
                  <span className="prop-label">UTS</span>
                  <span className="prop-value">{mat.properties.uts.toFixed(0)} MPa</span>
                </div>
                <div className="prop">
                  <span className="prop-label">ε</span>
                  <span className="prop-value">{mat.properties.elongation.toFixed(0)}%</span>
                </div>
              </div>
              
              <button 
                className="select-material-btn"
                onClick={() => onSelectMaterial(mat.id, mat.name)}
                disabled={disabled}
                style={{ backgroundColor: categoryColors[mat.category] || '#666' }}
              >
                {disabled ? 'Analyzing...' : 'Select & Analyze'}
              </button>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default MaterialSelector;
