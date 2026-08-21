import type { ColumnDef } from '@tanstack/react-table';
import type { EbayListing } from '../../../types/ebay';
import { money } from '../utils';
import { EbayTab, baseEbayColumns } from '../../../components/EbayTab';

const columns: ColumnDef<EbayListing>[] = [
  ...baseEbayColumns,
  { accessorKey: 'price', header: 'Price', size: 90,
    cell: ({ getValue }) => money(getValue<string>()) },
];

export function BuyItNow({ isActive }: { isActive: boolean }) {
  return (
    <EbayTab
      isActive={isActive}
      endpoint="api/ebay/buyitnows"
      refreshEndpoint="api/ebay/refresh_buyitnows"
      columns={columns}
      title="BUY IT NOW"
      storageKey="ebay_bin_results"
    />
  );
}