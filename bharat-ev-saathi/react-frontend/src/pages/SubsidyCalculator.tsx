import React, { useState, useEffect } from 'react';
import { Calculator, IndianRupee, Car, Bike, TrendingDown, Info, CheckCircle2 } from 'lucide-react';

interface SubsidyResult {
  fame_subsidy: number;
  state_subsidy: number;
  total_subsidy: number;
  final_price: number;
  savings_percentage: number;
  eligible: boolean;
  message: string;
}

interface State {
  name: string;
  subsidy_2w?: number;
  subsidy_3w?: number;
  subsidy_4w?: number;
}

const SubsidyCalculator = () => {
  const [vehicleType, setVehicleType] = useState<string>('2-Wheeler');
  const [batteryCapacity, setBatteryCapacity] = useState<string>('');
  const [vehiclePrice, setVehiclePrice] = useState<string>('');
  const [selectedState, setSelectedState] = useState<string>('');
  const [states, setStates] = useState<State[]>([]);
  const [result, setResult] = useState<SubsidyResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>('');

  useEffect(() => {
    fetchStates();
  }, []);

  const fetchStates = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/subsidy/states');
      const data = await response.json();
      setStates(data.states || []);
      if (data.states && data.states.length > 0) {
        setSelectedState(data.states[0].name);
      }
    } catch (err) {
      console.error('Error fetching states:', err);
      setStates([
        { name: 'Maharashtra', subsidy_2w: 10000, subsidy_3w: 20000, subsidy_4w: 100000 },
        { name: 'Delhi', subsidy_2w: 30000, subsidy_3w: 30000, subsidy_4w: 150000 },
        { name: 'Karnataka', subsidy_2w: 10000, subsidy_3w: 0, subsidy_4w: 0 },
        { name: 'Gujarat', subsidy_2w: 10000, subsidy_3w: 25000, subsidy_4w: 20000 }
      ]);
      setSelectedState('Maharashtra');
    }
  };

  const calculateSubsidy = async () => {
    if (!batteryCapacity || !vehiclePrice || !selectedState) {
      setError('Please fill all fields');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const response = await fetch('http://localhost:8000/api/subsidy/calculate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          vehicle_type: vehicleType,
          battery_capacity: parseFloat(batteryCapacity),
          vehicle_price: parseFloat(vehiclePrice),
          state: selectedState
        })
      });

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError('Failed to calculate subsidy. Please try again.');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const vehicleTypeInfo = {
    '2-Wheeler': {
      icon: Bike,
      fameAmount: '₹15,000',
      condition: 'Min 2kWh battery, Price ≤ ₹1.5L'
    },
    '3-Wheeler': {
      icon: Car,
      fameAmount: '₹10,000/kWh',
      condition: 'Up to ₹50,000 max'
    },
    '4-Wheeler': {
      icon: Car,
      fameAmount: '₹10,000/kWh',
      condition: 'Up to ₹1.5L max, Price ≤ ₹15L'
    }
  };

  const TypeIcon = vehicleTypeInfo[vehicleType as keyof typeof vehicleTypeInfo].icon;

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="container mx-auto px-4 max-w-6xl">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold mb-4 flex items-center justify-center gap-3">
            <Calculator className="text-green-600" size={40} />
            FAME-II Subsidy Calculator
          </h1>
          <p className="text-gray-600 text-lg">
            Calculate your electric vehicle subsidies under FAME-II scheme
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Input Form */}
          <div className="bg-white rounded-xl shadow-lg p-8">
            <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
              <IndianRupee className="text-blue-600" />
              Vehicle Details
            </h2>

            {/* Vehicle Type */}
            <div className="mb-6">
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                Vehicle Type
              </label>
              <div className="grid grid-cols-3 gap-3">
                {['2-Wheeler', '3-Wheeler', '4-Wheeler'].map((type) => (
                  <button
                    key={type}
                    onClick={() => setVehicleType(type)}
                    className={`py-3 px-4 rounded-lg font-semibold transition-all ${
                      vehicleType === type
                        ? 'bg-blue-600 text-white shadow-lg'
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                  >
                    {type}
                  </button>
                ))}
              </div>
            </div>

            {/* Battery Capacity */}
            <div className="mb-6">
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                Battery Capacity (kWh)
              </label>
              <input
                type="number"
                value={batteryCapacity}
                onChange={(e) => setBatteryCapacity(e.target.value)}
                placeholder="e.g., 3.5"
                step="0.1"
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            {/* Vehicle Price */}
            <div className="mb-6">
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                Vehicle Price (₹)
              </label>
              <input
                type="number"
                value={vehiclePrice}
                onChange={(e) => setVehiclePrice(e.target.value)}
                placeholder="e.g., 150000"
                step="1000"
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            {/* State Selection */}
            <div className="mb-6">
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                Select State
              </label>
              <select
                value={selectedState}
                onChange={(e) => setSelectedState(e.target.value)}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                {states.map((state) => (
                  <option key={state.name} value={state.name}>
                    {state.name}
                  </option>
                ))}
              </select>
            </div>

            {/* Calculate Button */}
            <button
              onClick={calculateSubsidy}
              disabled={loading}
              className="w-full bg-green-600 text-white py-4 rounded-lg font-bold text-lg hover:bg-green-700 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                  Calculating...
                </>
              ) : (
                <>
                  <Calculator size={20} />
                  Calculate Subsidy
                </>
              )}
            </button>

            {error && (
              <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
                {error}
              </div>
            )}
          </div>

          {/* Results */}
          <div>
            {/* FAME Info Card */}
            <div className="bg-linear-to-br from-blue-50 to-purple-50 rounded-xl p-6 mb-6 border border-blue-200">
              <div className="flex items-start gap-3 mb-4">
                <TypeIcon className="text-blue-600" size={32} />
                <div>
                  <h3 className="text-xl font-bold text-gray-900">
                    {vehicleType} - FAME-II Benefits
                  </h3>
                  <p className="text-gray-600 text-sm mt-1">
                    {vehicleTypeInfo[vehicleType as keyof typeof vehicleTypeInfo].condition}
                  </p>
                </div>
              </div>
              <div className="bg-white rounded-lg p-4">
                <div className="flex items-center justify-between">
                  <span className="text-gray-700 font-semibold">FAME-II Subsidy:</span>
                  <span className="text-2xl font-bold text-green-600">
                    {vehicleTypeInfo[vehicleType as keyof typeof vehicleTypeInfo].fameAmount}
                  </span>
                </div>
              </div>
            </div>

            {/* Results Card */}
            {result && (
              <div className="bg-white rounded-xl shadow-lg p-8">
                <div className="flex items-center gap-2 mb-6">
                  <CheckCircle2 className="text-green-600" size={28} />
                  <h3 className="text-2xl font-bold">Subsidy Breakdown</h3>
                </div>

                {result.eligible ? (
                  <div className="space-y-4">
                    {/* FAME Subsidy */}
                    <div className="flex justify-between items-center p-4 bg-blue-50 rounded-lg">
                      <span className="font-semibold text-gray-700">FAME-II Subsidy</span>
                      <span className="text-xl font-bold text-blue-600">
                        ₹{result.fame_subsidy.toLocaleString()}
                      </span>
                    </div>

                    {/* State Subsidy */}
                    <div className="flex justify-between items-center p-4 bg-green-50 rounded-lg">
                      <span className="font-semibold text-gray-700">State Subsidy ({selectedState})</span>
                      <span className="text-xl font-bold text-green-600">
                        ₹{result.state_subsidy.toLocaleString()}
                      </span>
                    </div>

                    {/* Total Subsidy */}
                    <div className="flex justify-between items-center p-4 bg-linear-to-r from-purple-100 to-pink-100 rounded-lg border-2 border-purple-300">
                      <span className="font-bold text-gray-800">Total Subsidy</span>
                      <span className="text-2xl font-bold text-purple-700">
                        ₹{result.total_subsidy.toLocaleString()}
                      </span>
                    </div>

                    <hr className="my-4" />

                    {/* Price Breakdown */}
                    <div className="space-y-3">
                      <div className="flex justify-between text-gray-700">
                        <span>Original Price:</span>
                        <span className="font-semibold">₹{parseFloat(vehiclePrice).toLocaleString()}</span>
                      </div>
                      <div className="flex justify-between text-gray-700">
                        <span>Total Discount:</span>
                        <span className="font-semibold text-red-600">
                          - ₹{result.total_subsidy.toLocaleString()}
                        </span>
                      </div>
                      <div className="flex justify-between items-center p-4 bg-linear-to-r from-green-500 to-green-600 rounded-lg text-white">
                        <span className="font-bold text-lg">Final Price:</span>
                        <span className="text-3xl font-bold">
                          ₹{result.final_price.toLocaleString()}
                        </span>
                      </div>
                    </div>

                    {/* Savings Badge */}
                    <div className="mt-6 text-center p-4 bg-linear-to-r from-yellow-50 to-orange-50 rounded-lg border border-orange-200">
                      <TrendingDown className="text-orange-600 mx-auto mb-2" size={32} />
                      <p className="text-sm text-gray-700 mb-1">You save</p>
                      <p className="text-3xl font-bold text-orange-600">
                        {result.savings_percentage.toFixed(1)}%
                      </p>
                      <p className="text-xs text-gray-600 mt-1">
                        ₹{result.total_subsidy.toLocaleString()} total savings
                      </p>
                    </div>
                  </div>
                ) : (
                  <div className="text-center p-6 bg-yellow-50 border border-yellow-200 rounded-lg">
                    <Info className="text-yellow-600 mx-auto mb-3" size={40} />
                    <p className="text-gray-700 font-semibold">{result.message}</p>
                  </div>
                )}
              </div>
            )}

            {!result && (
              <div className="bg-white rounded-xl shadow-lg p-8 text-center">
                <Calculator className="text-gray-300 mx-auto mb-4" size={60} />
                <p className="text-gray-500 text-lg">
                  Enter vehicle details and click calculate to see your subsidy breakdown
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Info Section */}
        <div className="mt-8 bg-white rounded-xl shadow-lg p-8">
          <h3 className="text-2xl font-bold mb-6 flex items-center gap-2">
            <Info className="text-blue-600" />
            About FAME-II Scheme
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="p-4 bg-blue-50 rounded-lg">
              <Bike className="text-blue-600 mb-3" size={32} />
              <h4 className="font-bold text-gray-900 mb-2">2-Wheelers</h4>
              <p className="text-sm text-gray-600">
                Fixed subsidy of ₹15,000 for vehicles with battery ≥2kWh and price ≤₹1.5 lakh
              </p>
            </div>
            <div className="p-4 bg-green-50 rounded-lg">
              <Car className="text-green-600 mb-3" size={32} />
              <h4 className="font-bold text-gray-900 mb-2">3-Wheelers</h4>
              <p className="text-sm text-gray-600">
                ₹10,000 per kWh battery capacity, maximum subsidy up to ₹50,000
              </p>
            </div>
            <div className="p-4 bg-purple-50 rounded-lg">
              <Car className="text-purple-600 mb-3" size={32} />
              <h4 className="font-bold text-gray-900 mb-2">4-Wheelers</h4>
              <p className="text-sm text-gray-600">
                ₹10,000 per kWh, max ₹1.5L for vehicles priced ≤₹15 lakh
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SubsidyCalculator;
