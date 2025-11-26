import React, { useState } from 'react';
import axios from 'axios';

/**
 * Material Clustering Component
 * Uses K-means clustering to group similar materials
 */
const ClusteringPanel = ({ materials, onClose }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [clusterData, setClusterData] = useState(null);
  const [nClusters, setNClusters] = useState(3);

  const handleCluster = async () => {
    if (!materials || materials.length < 2) {
      setError('Need at least 2 materials for clustering');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const materialsData = materials.map(m => ({
        name: m.name,
        ...m.results
      }));

      const response = await axios.post('/api/cluster-materials', {
        materials: materialsData,
        nClusters: Math.min(nClusters, materials.length)
      });

      setClusterData(response.data);

    } catch (err) {
      setError(err.response?.data?.error || 'Clustering failed');
      console.error('Clustering error:', err);
    } finally {
      setLoading(false);
    }
  };

  const clusterColors = ['#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#8b5cf6'];

  return (
    <div className="clustering-panel">
      <div className="clustering-header">
        <h3>🔮 Material Clustering</h3>
        {onClose && (
          <button className="close-btn" onClick={onClose}>✕</button>
        )}
      </div>

      <p className="clustering-desc">
        Group similar materials based on their mechanical properties using K-means clustering.
      </p>

      {error && (
        <div className="clustering-error">
          {error}
        </div>
      )}

      <div className="cluster-controls">
        <label>
          Number of Clusters:
          <input 
            type="range" 
            min="2" 
            max={Math.min(5, materials?.length || 2)}
            value={nClusters}
            onChange={(e) => setNClusters(parseInt(e.target.value))}
          />
          <span className="cluster-count">{nClusters}</span>
        </label>

        <button 
          className="cluster-btn"
          onClick={handleCluster}
          disabled={loading || !materials || materials.length < 2}
        >
          {loading ? 'Clustering...' : '🎯 Run Clustering'}
        </button>
      </div>

      {clusterData && (
        <div className="cluster-results">
          {/* Cluster Chart */}
          {clusterData.clusterChart && (
            <div className="cluster-chart">
              <img src={clusterData.clusterChart} alt="Material Clusters" />
            </div>
          )}

          {/* Cluster Summary */}
          <div className="cluster-summary">
            <h4>📊 Cluster Summary</h4>
            {clusterData.clusters?.map((cluster, idx) => (
              <div 
                key={idx} 
                className="cluster-group"
                style={{ borderLeftColor: clusterColors[idx % clusterColors.length] }}
              >
                <div className="cluster-title">
                  <span 
                    className="cluster-badge"
                    style={{ backgroundColor: clusterColors[idx % clusterColors.length] }}
                  >
                    Cluster {cluster.cluster}
                  </span>
                  <span className="cluster-count-badge">{cluster.count} materials</span>
                </div>
                <div className="cluster-materials">
                  {cluster.materials.map((name, i) => (
                    <span key={i} className="material-tag">{name}</span>
                  ))}
                </div>
              </div>
            ))}
          </div>

          {/* PCA Variance */}
          {clusterData.pcaVariance && (
            <div className="pca-info">
              <small>
                PCA explains {((clusterData.pcaVariance[0] + clusterData.pcaVariance[1]) * 100).toFixed(1)}% 
                of variance (PC1: {(clusterData.pcaVariance[0] * 100).toFixed(1)}%, 
                PC2: {(clusterData.pcaVariance[1] * 100).toFixed(1)}%)
              </small>
            </div>
          )}
        </div>
      )}

      {!clusterData && !loading && (
        <div className="cluster-placeholder">
          <p>🎯 Click "Run Clustering" to group materials by similarity</p>
          <p className="hint">Materials will be grouped based on: Young's Modulus, Yield Stress, UTS, Elongation, Toughness, Resilience</p>
        </div>
      )}
    </div>
  );
};

export default ClusteringPanel;
