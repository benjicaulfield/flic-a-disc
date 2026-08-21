import { Auction } from './tabs/Auction';
import { Browse } from './tabs/Browse';
import { BuyItNow } from './tabs/BuyItNow';
import { BySeller } from './tabs/BySeller';
import { Train } from './tabs/Train';

export interface TabDef {
  id: string
  label: string
  Component: React.ComponentType< { isActive: boolean }>
}

export const Tabs: TabDef[] = [
  { id: 'browse',        label: 'Browse',        Component: Browse },
  { id: 'train',         label: 'Train',         Component: Train },
  { id: 'by-seller',     label: 'By Seller',     Component: BySeller },
  { id: 'auction',       label: 'Auction',       Component: Auction },
  { id: 'buyitnow',      label: 'Buy It Now',    Component: BuyItNow },
]

