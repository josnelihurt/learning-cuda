type ReactAddSourceFabProps = {
  onClick: () => void;
};

export function ReactAddSourceFab({ onClick }: ReactAddSourceFabProps) {
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
