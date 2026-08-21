import type { ColumnDef } from '@tanstack/react-table';
import type { EbayListing } from '../../../types';
import { money } from '../utils';
import { EbayTab, baseEbayColumns } from '../../../components/EbayTab';

const columns: ColumnDef<EbayListing>[] = [
  ...baseEbayColumns,
  { accessorKey: 'price', header: 'Current Bid', size: 90,
    cell: ({ getValue }) => money(getValue<string>()) },
];

export function Auction({ isActive }: { isActive: boolean }) {
  return (
    <EbayTab
      isActive={isActive}
      endpoint="api/ebay/auctions"
      refreshEndpoint="api/ebay/refresh_auctions"
      columns={columns}
      title="AUCTIONS"
      storageKey="ebay_auction_results"
    />
  );
}