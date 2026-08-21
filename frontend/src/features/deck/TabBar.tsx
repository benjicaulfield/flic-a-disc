import type { TabDef } from './tabs';


interface TabBarProps {
  tabs: TabDef[];
  activeId: string;
  onSelect: (id: string) => void;
}

export function TabBar({ tabs, activeId, onSelect }: TabBarProps) {
  return (
    <div className="flex gap-1 mb-1">
      {tabs.map(({ id, label }) => (
        <button
          key={id}
          onClick={() => onSelect(id)}
          className={`w-28 py-1.5 text-sm rounded border font-semibold uppercase ${
            activeId === id
              ? 'bg-slate-700 border-slate-600 text-white shadow-inner'
              : 'bg-slate-500 border-slate-400 text-white hover:bg-slate-600'
          }`}
        >
          {label}
        </button>
      ))}
    </div>
  );
}