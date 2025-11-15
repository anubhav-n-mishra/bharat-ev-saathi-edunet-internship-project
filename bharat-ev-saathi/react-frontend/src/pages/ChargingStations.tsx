import React, { useState, useEffect } from 'react';
import { MapPin, Search, Zap, Navigation } from 'lucide-react';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

// Fix default marker icon issue with Leaflet in React
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

// Custom charging station icon
const chargingIcon = new L.Icon({
  iconUrl: 'data:image/svg+xml;base64,' + btoa(`
    <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/>
    </svg>
  `),
  iconSize: [32, 32],
  iconAnchor: [16, 32],
  popupAnchor: [0, -32],
});

interface ChargingStation {
  id: number;
  name: string;
  city: string;
  state: string;
  address?: string;
  latitude: number;
  longitude: number;
  operator?: string;
  charger_type?: string;
  num_chargers?: number;
}

const ChargingStations = () => {
  const [stations, setStations] = useState<ChargingStation[]>([]);
  const [filteredStations, setFilteredStations] = useState<ChargingStation[]>([]);
  const [cities, setCities] = useState<string[]>([]);
  const [selectedCity, setSelectedCity] = useState<string>('');
  const [searchQuery, setSearchQuery] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [mapCenter, setMapCenter] = useState<[number, number]>([20.5937, 78.9629]); // India center
  const [mapZoom, setMapZoom] = useState(5);

  useEffect(() => {
    fetchChargingStations();
  }, []);

  useEffect(() => {
    filterStations();
  }, [selectedCity, searchQuery, stations]);

  const fetchChargingStations = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/charging-stations');
      if (response.ok) {
        const data = await response.json();
        setStations(data.stations || []);
        
        // Extract unique cities
        const uniqueCities = [...new Set(data.stations.map((s: ChargingStation) => s.city))].sort();
        setCities(uniqueCities);
      }
    } catch (error) {
      console.error('Error fetching charging stations:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const filterStations = () => {
    let filtered = stations;

    // Filter by city
    if (selectedCity) {
      filtered = filtered.filter(s => s.city === selectedCity);
    }

    // Filter by search query
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(s =>
        s.name.toLowerCase().includes(query) ||
        s.city.toLowerCase().includes(query) ||
        s.state.toLowerCase().includes(query) ||
        (s.address && s.address.toLowerCase().includes(query))
      );
    }

    setFilteredStations(filtered);

    // Update map center if city is selected
    if (filtered.length > 0) {
      const avgLat = filtered.reduce((sum, s) => sum + s.latitude, 0) / filtered.length;
      const avgLng = filtered.reduce((sum, s) => sum + s.longitude, 0) / filtered.length;
      setMapCenter([avgLat, avgLng]);
      setMapZoom(selectedCity ? 12 : 10);
    }
  };

  const handleCityChange = (city: string) => {
    setSelectedCity(city);
    setSearchQuery('');
  };

  const resetFilters = () => {
    setSelectedCity('');
    setSearchQuery('');
    setMapCenter([20.5937, 78.9629]);
    setMapZoom(5);
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <Zap className="w-16 h-16 text-blue-600 mx-auto mb-4 animate-pulse" />
          <p className="text-xl text-gray-600">Loading charging stations...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="container mx-auto px-4">
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-4 flex items-center gap-3">
            <Zap className="text-blue-600" size={40} />
            EV Charging Stations
          </h1>
          <p className="text-gray-600 text-lg">
            Find {stations.length}+ charging stations across India with interactive map
          </p>
        </div>

        {/* Filters */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* City Dropdown */}
            <div>
              <label className="block text-sm font-semibold mb-2 text-gray-700">
                Filter by City
              </label>
              <select
                value={selectedCity}
                onChange={(e) => handleCityChange(e.target.value)}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="">All Cities ({cities.length})</option>
                {cities.map(city => (
                  <option key={city} value={city}>
                    {city} ({stations.filter(s => s.city === city).length})
                  </option>
                ))}
              </select>
            </div>

            {/* Search */}
            <div>
              <label className="block text-sm font-semibold mb-2 text-gray-700">
                Search Stations
              </label>
              <div className="relative">
                <Search className="absolute left-3 top-3.5 text-gray-400" size={20} />
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Search by name, city, or location..."
                  className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
            </div>

            {/* Stats & Reset */}
            <div className="flex items-end">
              <div className="w-full">
                <div className="bg-blue-50 rounded-lg p-4 mb-2">
                  <p className="text-sm text-gray-600">Showing</p>
                  <p className="text-2xl font-bold text-blue-600">
                    {filteredStations.length}
                  </p>
                  <p className="text-sm text-gray-600">stations</p>
                </div>
                {(selectedCity || searchQuery) && (
                  <button
                    onClick={resetFilters}
                    className="w-full px-4 py-2 bg-gray-200 hover:bg-gray-300 rounded-lg transition-colors text-sm font-semibold"
                  >
                    Reset Filters
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Map and List */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Map */}
          <div className="bg-white rounded-xl shadow-lg overflow-hidden h-[600px]">
            <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white p-4">
              <h3 className="font-semibold flex items-center gap-2">
                <Navigation size={20} />
                Interactive Map
              </h3>
            </div>
            <MapContainer
              center={mapCenter}
              zoom={mapZoom}
              style={{ height: '550px', width: '100%' }}
              key={`${mapCenter[0]}-${mapCenter[1]}-${mapZoom}`}
            >
              <TileLayer
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              />
              {filteredStations.map((station) => (
                <Marker
                  key={station.id}
                  position={[station.latitude, station.longitude]}
                  icon={chargingIcon}
                >
                  <Popup>
                    <div className="p-2">
                      <h4 className="font-bold text-blue-600 mb-2">{station.name}</h4>
                      <p className="text-sm mb-1">
                        <strong>City:</strong> {station.city}
                      </p>
                      {station.address && (
                        <p className="text-sm mb-1">
                          <strong>Address:</strong> {station.address}
                        </p>
                      )}
                      {station.operator && (
                        <p className="text-sm mb-1">
                          <strong>Operator:</strong> {station.operator}
                        </p>
                      )}
                      {station.charger_type && (
                        <p className="text-sm">
                          <strong>Type:</strong> {station.charger_type}
                        </p>
                      )}
                    </div>
                  </Popup>
                </Marker>
              ))}
            </MapContainer>
          </div>

          {/* Station List */}
          <div className="bg-white rounded-xl shadow-lg overflow-hidden h-[600px] flex flex-col">
            <div className="bg-gradient-to-r from-green-600 to-blue-600 text-white p-4">
              <h3 className="font-semibold flex items-center gap-2">
                <MapPin size={20} />
                Station List
              </h3>
            </div>
            <div className="flex-1 overflow-y-auto p-4 space-y-3">
              {filteredStations.length === 0 ? (
                <div className="text-center py-12">
                  <MapPin className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                  <p className="text-gray-500">No charging stations found</p>
                  <p className="text-sm text-gray-400 mt-2">Try adjusting your filters</p>
                </div>
              ) : (
                filteredStations.map((station) => (
                  <div
                    key={station.id}
                    className="border border-gray-200 rounded-lg p-4 hover:border-blue-400 hover:shadow-md transition-all cursor-pointer"
                  >
                    <div className="flex items-start gap-3">
                      <div className="flex-shrink-0 w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center">
                        <Zap className="text-blue-600" size={20} />
                      </div>
                      <div className="flex-1">
                        <h4 className="font-semibold text-gray-800 mb-1">{station.name}</h4>
                        <p className="text-sm text-gray-600 mb-2">
                          <MapPin className="inline w-4 h-4 mr-1" />
                          {station.city}, {station.state}
                        </p>
                        {station.address && (
                          <p className="text-xs text-gray-500 mb-2">{station.address}</p>
                        )}
                        <div className="flex gap-4 text-xs">
                          {station.operator && (
                            <span className="bg-green-50 text-green-700 px-2 py-1 rounded">
                              {station.operator}
                            </span>
                          )}
                          {station.charger_type && (
                            <span className="bg-blue-50 text-blue-700 px-2 py-1 rounded">
                              {station.charger_type}
                            </span>
                          )}
                          {station.num_chargers && (
                            <span className="bg-purple-50 text-purple-700 px-2 py-1 rounded">
                              {station.num_chargers} chargers
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>

        {/* Quick Stats */}
        <div className="mt-8 grid grid-cols-1 md:grid-cols-4 gap-6">
          <div className="bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl p-6 text-white">
            <Zap className="mb-3" size={32} />
            <p className="text-3xl font-bold">{stations.length}</p>
            <p className="text-blue-100">Total Stations</p>
          </div>
          <div className="bg-gradient-to-br from-green-500 to-green-600 rounded-xl p-6 text-white">
            <MapPin className="mb-3" size={32} />
            <p className="text-3xl font-bold">{cities.length}</p>
            <p className="text-green-100">Cities Covered</p>
          </div>
          <div className="bg-gradient-to-br from-purple-500 to-purple-600 rounded-xl p-6 text-white">
            <Navigation className="mb-3" size={32} />
            <p className="text-3xl font-bold">
              {[...new Set(stations.map(s => s.state))].length}
            </p>
            <p className="text-purple-100">States</p>
          </div>
          <div className="bg-gradient-to-br from-orange-500 to-orange-600 rounded-xl p-6 text-white">
            <Search className="mb-3" size={32} />
            <p className="text-3xl font-bold">{filteredStations.length}</p>
            <p className="text-orange-100">Filtered Results</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChargingStations;
