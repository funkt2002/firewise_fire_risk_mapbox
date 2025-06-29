import React from 'react';
import { useAppState } from '../../hooks/useAppState';

function CorrelationModal() {
  const { actions } = useAppState();
  
  return (
    <div className="modal-overlay" onClick={() => actions.toggleModal('correlation')}>
      <div className="modal-content" onClick={e => e.stopPropagation()}>
        <h2>Correlation Analysis</h2>
        <p>Correlation modal placeholder - to be implemented</p>
        <button onClick={() => actions.toggleModal('correlation')}>Close</button>
      </div>
    </div>
  );
}

export default CorrelationModal;