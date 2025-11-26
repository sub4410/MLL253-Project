import React, { useState, useEffect } from 'react';

/**
 * Interactive Tutorial / Walkthrough Component
 */
const Tutorial = ({ onComplete, isFirstVisit }) => {
  const [step, setStep] = useState(0);
  const [visible, setVisible] = useState(isFirstVisit);

  const steps = [
    {
      title: "Welcome to Stress-Strain Analyzer! 🔬",
      content: "This tool helps you analyze mechanical properties of materials from stress-strain data. Let's take a quick tour!",
      icon: "👋"
    },
    {
      title: "Upload Your Data 📤",
      content: "Start by uploading a CSV file with strain and stress columns, or choose from our material library with real experimental data.",
      icon: "📁",
      highlight: ".input-panel"
    },
    {
      title: "Choose Analysis Method 🧮",
      content: "Select between Mathematical (0.2% offset yield) or Machine Learning (regression-based) analysis approaches.",
      icon: "⚙️",
      highlight: ".parameters-form"
    },
    {
      title: "Compare Multiple Materials 📊",
      content: "Add up to 10 materials to compare their properties side-by-side with interactive charts.",
      icon: "📈",
      highlight: ".material-list"
    },
    {
      title: "Interactive Charts 📉",
      content: "Hover over charts to see exact values. Toggle between interactive Plotly charts and static images.",
      icon: "🖱️",
      highlight: ".graph-container"
    },
    {
      title: "Export & Share 💾",
      content: "Export your results as PDF reports, Excel spreadsheets, or JSON. Save sessions to continue later!",
      icon: "📤",
      highlight: ".export-panel"
    },
    {
      title: "Material Clustering 🔮",
      content: "Use ML clustering to find similar materials based on their mechanical properties.",
      icon: "🎯",
      highlight: ".clustering-panel"
    },
    {
      title: "You're Ready! 🎉",
      content: "Start by uploading a file or selecting a material from the library. Happy analyzing!",
      icon: "🚀"
    }
  ];

  useEffect(() => {
    // Highlight current element
    const currentStep = steps[step];
    if (currentStep.highlight) {
      const element = document.querySelector(currentStep.highlight);
      if (element) {
        element.classList.add('tutorial-highlight');
        return () => element.classList.remove('tutorial-highlight');
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [step]);

  const handleNext = () => {
    if (step < steps.length - 1) {
      setStep(step + 1);
    } else {
      handleComplete();
    }
  };

  const handlePrev = () => {
    if (step > 0) {
      setStep(step - 1);
    }
  };

  const handleComplete = () => {
    setVisible(false);
    localStorage.setItem('tutorialCompleted', 'true');
    if (onComplete) onComplete();
  };

  const handleSkip = () => {
    handleComplete();
  };

  if (!visible) {
    return (
      <button 
        className="tutorial-trigger"
        onClick={() => setVisible(true)}
        title="Show Tutorial"
      >
        ❓
      </button>
    );
  }

  const currentStep = steps[step];

  return (
    <div className="tutorial-overlay">
      <div className="tutorial-modal">
        <div className="tutorial-icon">{currentStep.icon}</div>
        
        <h2 className="tutorial-title">{currentStep.title}</h2>
        
        <p className="tutorial-content">{currentStep.content}</p>

        <div className="tutorial-progress">
          {steps.map((_, idx) => (
            <span 
              key={idx} 
              className={`progress-dot ${idx === step ? 'active' : ''} ${idx < step ? 'completed' : ''}`}
              onClick={() => setStep(idx)}
            />
          ))}
        </div>

        <div className="tutorial-actions">
          {step > 0 && (
            <button className="tutorial-btn secondary" onClick={handlePrev}>
              ← Back
            </button>
          )}
          
          <button className="tutorial-btn skip" onClick={handleSkip}>
            Skip Tutorial
          </button>
          
          <button className="tutorial-btn primary" onClick={handleNext}>
            {step === steps.length - 1 ? "Get Started! 🚀" : "Next →"}
          </button>
        </div>
      </div>
    </div>
  );
};

export default Tutorial;
