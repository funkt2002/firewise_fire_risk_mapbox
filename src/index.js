import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

// Wait for the page to load before mounting React
window.addEventListener('DOMContentLoaded', () => {
  // Find or create the React mount point
  let mountPoint = document.getElementById('react-root');
  
  if (!mountPoint) {
    // Create mount point in the control panel area
    const controlPanel = document.querySelector('.control-panel');
    if (controlPanel) {
      mountPoint = document.createElement('div');
      mountPoint.id = 'react-root';
      // Replace the control panel content with React
      controlPanel.innerHTML = '';
      controlPanel.appendChild(mountPoint);
    } else {
      console.error('Could not find control panel to mount React app');
      return;
    }
  }
  
  // Mount the React app
  const root = ReactDOM.createRoot(mountPoint);
  root.render(
    <React.StrictMode>
      <App />
    </React.StrictMode>
  );
  
  console.log('React app mounted successfully');
});