import React from 'react';

type AddSourceFabProps = {
  onClick: () => void;
};

export function AddSourceFab({ onClick }: AddSourceFabProps) {
  return (
    <button
      type="button"
      className="react-add-source-fab"
      onClick={onClick}
      data-testid="add-input-fab"
    >
      <span className="fab-icon">+</span>
      <span className="fab-text">Add Input</span>
    </button>
  );
}
