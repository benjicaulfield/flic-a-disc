import type { ColumnDef } from '@tanstack/react-table';
import type { EbayListing } from '../../types/ebay';
import type { DiscogsRecord } from '../../types/discogs';

const joinList = (v: unknown) => (Array.isArray(v) ? v.join(', ') : '') || 'N/A';
const money = (v: number | null | undefined) =>
  v == null ? 'N/A' : `$${v.toFixed(2)}`;

export const discogsColumns: ColumnDef<DiscogsRecord>[] = [
  { accessorKey: 'artist', header: 'Artist', size: 200 },
  { accessorKey: 'title',  header: 'Title',  size: 220 },
  { accessorKey: 'format', header: 'Format', size: 70,  cell: ({ getValue }) => joinList(getValue()) },
  { accessorKey: 'label',  header: 'Label',  size: 160 },
  { accessorKey: 'year',   header: 'Year',   size: 60,  cell: ({ getValue }) => getValue<number | null>() ?? 'N/A' },
  { accessorKey: 'genres', header: 'Genre',  size: 120, cell: ({ getValue }) => joinList(getValue()) },
  { accessorKey: 'styles', header: 'Style',  size: 120, cell: ({ getValue }) => joinList(getValue()) },
  { accessorKey: 'wants',  header: 'Wants',  size: 65 },
  { accessorKey: 'haves',  header: 'Haves',  size: 65 },
  { accessorKey: 'suggested_price', header: 'Sugg.$', size: 80, cell: ({ getValue }) => money(getValue<number>()) },
]

export const basicEbayColumns: ColumnDef<EbayListing>[] = [
  { accessorKey: 'ebay_title', header: 'Title', size: 200 },
  { accessorKey: 'price',      header: 'Price', size: 200 },
]

export const enrichedEbayColumns: ColumnDef<EbayListing>[] = [
  { accessorKey: 'listing.ebay_title', header: 'Ebay Title', size: 200 },
  { accessorKey: 'artist', header: 'Artist', size: 200 },
  { accessorKey: 'title',  header: 'Title',  size: 220 },
  { accessorKey: 'format', header: 'Format', size: 70,  cell: ({ getValue }) => joinList(getValue()) },
  { accessorKey: 'year',   header: 'Year',   size: 60,  cell: ({ getValue }) => getValue<number | null>() ?? 'N/A' },
  { accessorKey: 'media_condition',  header: 'Condition',  size: 220 },
  { accessorKey: 'genres', header: 'Genre',  size: 120, cell: ({ getValue }) => joinList(getValue()) },
  { accessorKey: 'styles', header: 'Style',  size: 120, cell: ({ getValue }) => joinList(getValue()) },
  { accessorKey: 'price',      header: 'Price', size: 200 },

]