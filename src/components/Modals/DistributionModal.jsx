import React from 'react';
import { useAppState } from '../../hooks/useAppState';

function DistributionModal() {
  const { actions } = useAppState();
  
  return (
    <div className="modal-overlay" onClick={() => actions.toggleModal('distribution')}>
      <div className="modal-content" onClick={e => e.stopPropagation()}>
        <h2>Distribution Plots</h2>
        <p>Distribution modal placeholder - to be implemented</p>
        <button onClick={() => actions.toggleModal('distribution')}>Close</button>
      </div>
    </div>
  );
}

export default DistributionModal;