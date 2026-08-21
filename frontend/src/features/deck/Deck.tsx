import { useState } from 'react';
import { TabBar } from './TabBar';
import { Tabs } from './tabs';


export default function Deck() {
  const [activeId, setActiveId] = useState(Tabs[0].id);

  return (
    <div className="flex flex-col h-screen">
      <div className="flex-1 min-h-0 p-2 flex flex-col">
        <TabBar tabs={Tabs} activeId={activeId} onSelect={setActiveId} />
        {Tabs.map(({ id, Component }) => (
          <div 
            key={id}
            className="flex-1 min-h-0 flex flex-col"
            style={{ display: id === activeId ? 'flex' : 'none' }}
          >
            <Component isActive={id === activeId} />
          </div>
        ))}
      </div>
    </div>
  );
}
