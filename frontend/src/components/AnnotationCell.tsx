interface AnnotationCellProps {
  dbValue: boolean | null | undefined;
  dbEvaluated: boolean | null | undefined;
  override: boolean | undefined;
  onToggle: () => void;
}

export function AnnotationCell({ dbValue, dbEvaluated, override, onToggle }: AnnotationCellProps) {
  if (override === undefined && dbEvaluated && !dbValue) {
    return (
      <button
        type="button"
        onClick={onToggle}
        title="Evaluated — not a match. Click to change."
        className="h-4 w-4 flex items-center justify-center text-red-400 hover:text-red-500 text-sm leading-none"
      >
        ✕
      </button>
    );
  }

  return (
    <input
      type="checkbox"
      className="h-4 w-4"
      checked={override ?? dbValue ?? false}
      onChange={onToggle}
    />
  );
}
