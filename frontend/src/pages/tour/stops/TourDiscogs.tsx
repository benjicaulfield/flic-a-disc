const mockRecords = [
  { id: 1, artist: 'Arthur Russell', title: 'World Of Echo', year: 1986, label: 'Rough Trade', wants: 4821, haves: 1203, genres: ['Electronic', 'Folk'], styles: ['Experimental', 'Ambient'], suggested_price: '$42.00', selected: true },
  { id: 2, artist: 'Broadcast', title: 'Haha Sound', year: 2003, label: 'Warp', wants: 3102, haves: 890, genres: ['Electronic', 'Pop'], styles: ['Psychedelic', 'Indie Pop'], suggested_price: '$28.00', selected: true },
  { id: 3, artist: 'The Necks', title: 'Drive By', year: 2018, label: 'ReR Megacorp', wants: 1044, haves: 312, genres: ['Jazz'], styles: ['Free Improvisation', 'Ambient'], suggested_price: '$35.00', selected: false },
  { id: 4, artist: 'Grouper', title: 'Dragging A Dead Deer Up A Hill', year: 2008, label: 'Type', wants: 5930, haves: 2100, genres: ['Electronic', 'Pop'], styles: ['Shoegaze', 'Ambient'], suggested_price: '$55.00', selected: true },
  { id: 5, artist: 'Fennesz', title: 'Endless Summer', year: 2001, label: 'Mego', wants: 6204, haves: 1800, genres: ['Electronic'], styles: ['Glitch', 'Ambient'], suggested_price: '$48.00', selected: false },
  { id: 6, artist: 'Oren Ambarchi', title: 'Hubris', year: 2016, label: 'Editions Mego', wants: 890, haves: 245, genres: ['Electronic', 'Rock'], styles: ['Drone', 'Krautrock'], suggested_price: '$30.00', selected: true },
  { id: 7, artist: 'Loren Connors', title: 'Blues', year: 1995, label: 'Temporary Residence', wants: 720, haves: 188, genres: ['Rock', 'Blues'], styles: ['Experimental', 'Outsider'], suggested_price: '$65.00', selected: false },
  { id: 8, artist: 'Pauline Oliveros', title: 'Deep Listening', year: 1989, label: 'New Albion', wants: 2340, haves: 610, genres: ['Classical', 'Electronic'], styles: ['Minimalism', 'Drone'], suggested_price: '$38.00', selected: true },
  { id: 9, artist: 'Bill Orcutt', title: 'A History Of Every One', year: 2012, label: 'Palilalia', wants: 540, haves: 143, genres: ['Rock', 'Blues'], styles: ['Free Improvisation', 'Outsider'], suggested_price: '$44.00', selected: false },
  { id: 10, artist: 'Yellow Swans', title: 'At All Ends', year: 2008, label: 'Not Not Fun', wants: 1100, haves: 289, genres: ['Electronic', 'Rock'], styles: ['Noise', 'Drone'], suggested_price: '$70.00', selected: true },
  { id: 11, artist: 'Jandek', title: 'Ready For The House', year: 1978, label: 'Corwood', wants: 1820, haves: 490, genres: ['Rock', 'Blues'], styles: ['Outsider', 'Folk'], suggested_price: '$90.00', selected: false },
  { id: 12, artist: 'The Dead C', title: 'Trapdoor Fucking Exit', year: 1990, label: 'Siltbreeze', wants: 980, haves: 220, genres: ['Rock'], styles: ['Noise Rock', 'Drone'], suggested_price: '$120.00', selected: true },
];

export default function TourDiscogs() {
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
      {/* Header */}
      <div className="bg-white rounded-lg shadow-sm p-6 mb-6">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-4">
          <h1 className="text-3xl font-bold text-gray-900">Discogs Keepers</h1>
          <div className="mt-2 sm:mt-0 text-sm text-gray-600">
            Total labeled: <span className="font-semibold text-blue-600">3,240 / 14,802</span>
          </div>
        </div>

        <div className="flex items-center justify-between">
          <button className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md opacity-50 cursor-not-allowed">
            ← Previous
          </button>
          <span className="text-sm text-gray-700">Page <span className="font-medium">1</span> of <span className="font-medium">3</span></span>
          <button className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50">
            Next →
          </button>
        </div>
      </div>

      {/* Table */}
      <div className="bg-white rounded-lg shadow-sm overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full table-fixed divide-y divide-gray-200 text-xs">
            <thead className="bg-gray-50">
              <tr>
                <th className="w-12 px-2 py-2 text-left font-medium text-gray-500 uppercase">✓</th>
                <th className="w-32 px-2 py-2 text-left font-medium text-gray-500 uppercase">Artist</th>
                <th className="w-32 px-2 py-2 text-left font-medium text-gray-500 uppercase">Title</th>
                <th className="w-12 px-2 py-2 text-left font-medium text-gray-500 uppercase">Year</th>
                <th className="w-28 px-2 py-2 text-left font-medium text-gray-500 uppercase">Label</th>
                <th className="w-14 px-2 py-2 text-center font-medium text-gray-500 uppercase">Want</th>
                <th className="w-14 px-2 py-2 text-center font-medium text-gray-500 uppercase">Have</th>
                <th className="w-20 px-2 py-2 text-left font-medium text-gray-500 uppercase">Genre</th>
                <th className="w-20 px-2 py-2 text-left font-medium text-gray-500 uppercase">Style</th>
                <th className="w-20 px-2 py-2 text-right font-medium text-gray-500 uppercase">Sugg</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {mockRecords.map((record, index) => (
                <tr key={record.id} className={index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                  <td className="px-2 py-2">
                    <div className={`w-6 h-6 rounded border flex items-center justify-center text-xs ${
                      record.selected
                        ? 'bg-green-500 border-green-500 text-white'
                        : 'bg-white border-gray-300'
                    }`}>
                      {record.selected && '✓'}
                    </div>
                  </td>
                  <td className="px-2 py-2 font-medium text-gray-900 truncate">{record.artist}</td>
                  <td className="px-2 py-2 text-gray-700 truncate">{record.title}</td>
                  <td className="px-2 py-2 text-gray-700">{record.year}</td>
                  <td className="px-2 py-2 text-gray-700 truncate">{record.label}</td>
                  <td className="px-2 py-2 text-center">
                    <span className="px-1 py-0.5 rounded text-xs bg-blue-100 text-blue-800">{record.wants.toLocaleString()}</span>
                  </td>
                  <td className="px-2 py-2 text-center">
                    <span className="px-1 py-0.5 rounded text-xs bg-gray-100 text-gray-800">{record.haves.toLocaleString()}</span>
                  </td>
                  <td className="px-2 py-2 text-gray-700 truncate">{record.genres[0]}</td>
                  <td className="px-2 py-2 text-gray-700 truncate">{record.styles[0]}</td>
                  <td className="px-2 py-2 text-gray-500 text-right">{record.suggested_price}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="mt-6 flex justify-center">
        <button className="px-8 py-3 rounded-lg text-white font-medium bg-blue-600">
          Submit Page
        </button>
      </div>
    </div>
  );
}