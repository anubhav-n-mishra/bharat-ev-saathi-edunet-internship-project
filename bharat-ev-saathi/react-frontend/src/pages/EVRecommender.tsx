import { useState } from 'react';
import { Target, Zap, Brain } from 'lucide-react';

interface EVRecommendation {
  rank: number;
  brand: string;
  model: string;
  price: number;
  range: number;
  battery_kwh: number;
  efficiency: number;
  top_speed: number;
  charging_time: number;
  ml_confidence: number;
  total_score: number;
  ml_score: number;
  range_score: number;
  price_score: number;
  efficiency_score: number;
  fame_eligible: string;
  subsidy: number;
  user_rating: number;
  type: string;
  segment: string;
  predicted_segment: string;
  reasons: string[];
}

const EVRecommender = () => {
  const [budget, setBudget] = useState(1500000);
  const [dailyKm, setDailyKm] = useState(50);
  const [vehicleType, setVehicleType] = useState('4-Wheeler');
  const [topN, setTopN] = useState(5);
  const [loading, setLoading] = useState(false);
  const [recommendations, setRecommendations] = useState<EVRecommendation[]>([]);
  const [error, setError] = useState('');

  const handleGetRecommendations = async () => {
    setLoading(true);
    setError('');
    
    try {
      const response = await fetch('http://localhost:8000/api/recommendations', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          budget,
          daily_km: dailyKm,
          vehicle_type: vehicleType,
          top_n: topN,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to fetch recommendations');
      }

      const data = await response.json();
      setRecommendations(data.recommendations || []);
      
      if (data.recommendations.length === 0) {
        setError(`No EVs found matching your criteria. Try increasing your budget or changing vehicle type.`);
      }
    } catch (err) {
      setError('Unable to connect to the backend server. Please ensure the FastAPI server is running on port 8000.');
      console.error('Error fetching recommendations:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="container mx-auto px-4">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl md:text-5xl font-bold mb-4">
            <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              Smart EV Recommender
            </span>
          </h1>
          <p className="text-xl text-gray-600 mb-2">AI-Powered Recommendations using Production ML Model</p>
          <p className="text-gray-500">Get personalized EV recommendations based on your budget, usage, and preferences</p>
        </div>

        {/* Model Info */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 max-w-4xl mx-auto mb-8">
          <div className="card p-4 text-center">
            <div className="text-sm text-gray-600 mb-1">Model Type</div>
            <div className="text-lg font-bold">Ensemble</div>
          </div>
          <div className="card p-4 text-center">
            <div className="text-sm text-gray-600 mb-1">CV Accuracy</div>
            <div className="text-lg font-bold text-green-600">72.06%</div>
          </div>
          <div className="card p-4 text-center">
            <div className="text-sm text-gray-600 mb-1">Features</div>
            <div className="text-lg font-bold">15</div>
          </div>
          <div className="card p-4 text-center">
            <div className="text-sm text-gray-600 mb-1">Algorithms</div>
            <div className="text-lg font-bold">RF+GB+RF</div>
          </div>
        </div>

        <div className="grid lg:grid-cols-4 gap-8">
          {/* Sidebar - Filters */}
          <div className="lg:col-span-1">
            <div className="card p-6 sticky top-20">
              <h2 className="text-xl font-bold mb-6 flex items-center">
                <Target className="w-5 h-5 mr-2 text-blue-600" />
                Your Requirements
              </h2>
              
              <div className="space-y-6">
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">
                    💰 Maximum Budget (₹)
                  </label>
                  <input
                    type="number"
                    value={budget}
                    onChange={(e) => setBudget(Number(e.target.value))}
                    min={50000}
                    max={10000000}
                    step={50000}
                    className="input-field"
                  />
                  <p className="text-xs text-gray-500 mt-1">₹{(budget/100000).toFixed(1)} Lakhs</p>
                </div>

                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">
                    📏 Daily Driving (km)
                  </label>
                  <input
                    type="number"
                    value={dailyKm}
                    onChange={(e) => setDailyKm(Number(e.target.value))}
                    min={5}
                    max={300}
                    step={5}
                    className="input-field"
                  />
                </div>

                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">
                    🚗 Vehicle Type
                  </label>
                  <select
                    value={vehicleType}
                    onChange={(e) => setVehicleType(e.target.value)}
                    className="input-field"
                  >
                    <option value="2-Wheeler">2-Wheeler</option>
                    <option value="3-Wheeler">3-Wheeler</option>
                    <option value="4-Wheeler">4-Wheeler</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">
                    📊 Number of Recommendations
                  </label>
                  <input
                    type="range"
                    value={topN}
                    onChange={(e) => setTopN(Number(e.target.value))}
                    min={3}
                    max={10}
                    className="w-full"
                  />
                  <p className="text-sm text-gray-600 text-center mt-1">{topN} vehicles</p>
                </div>

                <button
                  onClick={handleGetRecommendations}
                  disabled={loading}
                  className="btn-primary w-full flex items-center justify-center space-x-2"
                >
                  {loading ? (
                    <>
                      <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                      <span>Analyzing...</span>
                    </>
                  ) : (
                    <>
                      <Zap className="w-5 h-5" />
                      <span>Get AI Recommendations</span>
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>

          {/* Main Content - Recommendations */}
          <div className="lg:col-span-3">
            {error && (
              <div className="bg-red-50 border border-red-200 text-red-800 px-6 py-4 rounded-lg mb-6">
                <p className="font-semibold">⚠️ {error}</p>
              </div>
            )}

            {recommendations.length > 0 && (
              <div>
                <h2 className="text-2xl font-bold mb-6">
                  🎯 Top {recommendations.length} Recommendations for You
                </h2>

                <div className="space-y-6">
                  {recommendations.map((rec) => (
                    <div key={rec.rank} className="card p-6 hover:shadow-2xl transition-all">
                      {/* Header */}
                      <div className="flex flex-wrap items-start justify-between mb-4">
                        <div>
                          <h3 className="text-2xl font-bold text-gray-900 mb-2">
                            #{rec.rank} · {rec.brand} {rec.model}
                          </h3>
                          <div className="flex flex-wrap gap-2">
                            <span className="badge badge-success">
                              🤖 ML Confidence: {rec.ml_confidence}%
                            </span>
                            <span className="badge badge-info">
                              📊 Overall Score: {rec.total_score.toFixed(1)}/100
                            </span>
                          </div>
                        </div>
                      </div>

                      {/* Key Specs */}
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                        <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                          <div className="text-xs text-gray-600 uppercase font-semibold mb-1">💰 Price</div>
                          <div className="text-lg font-bold text-gray-900">₹{rec.price.toLocaleString()}</div>
                          {rec.fame_eligible === 'Yes' && (
                            <div className="text-xs text-green-600 mt-1">💚 FAME: -₹{rec.subsidy.toLocaleString()}</div>
                          )}
                        </div>

                        <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                          <div className="text-xs text-gray-600 uppercase font-semibold mb-1">🔋 Range</div>
                          <div className="text-lg font-bold text-gray-900">{rec.range} km</div>
                          <div className="text-xs text-gray-600 mt-1">⚡ {rec.efficiency.toFixed(1)} km/kWh</div>
                        </div>

                        <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                          <div className="text-xs text-gray-600 uppercase font-semibold mb-1">🔌 Battery</div>
                          <div className="text-lg font-bold text-gray-900">{rec.battery_kwh} kWh</div>
                          <div className="text-xs text-gray-600 mt-1">⏱️ {rec.charging_time}h charge</div>
                        </div>

                        <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                          <div className="text-xs text-gray-600 uppercase font-semibold mb-1">🏎️ Performance</div>
                          <div className="text-lg font-bold text-gray-900">{rec.top_speed} km/h</div>
                          {rec.user_rating > 0 && (
                            <div className="text-xs text-gray-600 mt-1">⭐ {rec.user_rating}/5 rating</div>
                          )}
                        </div>
                      </div>

                      {/* Expandable Details */}
                      <details className="mt-4">
                        <summary className="cursor-pointer text-blue-600 font-semibold hover:text-blue-700">
                          📊 View Detailed Analysis →
                        </summary>
                        
                        <div className="mt-4 space-y-4">
                          {/* Score Breakdown */}
                          <div>
                            <h4 className="font-bold text-gray-900 mb-3">🎯 Score Breakdown</h4>
                            <div className="grid md:grid-cols-2 gap-4">
                              <div className="bg-blue-50 border-l-4 border-blue-500 p-4 rounded">
                                <div className="font-semibold">🤖 ML Score: {rec.ml_score.toFixed(1)}/50</div>
                                <div className="text-sm text-gray-600">AI confidence in this recommendation</div>
                              </div>
                              <div className="bg-green-50 border-l-4 border-green-500 p-4 rounded">
                                <div className="font-semibold">📏 Range Score: {rec.range_score.toFixed(1)}/25</div>
                                <div className="text-sm text-gray-600">Meets your {dailyKm} km/day needs</div>
                              </div>
                              <div className="bg-purple-50 border-l-4 border-purple-500 p-4 rounded">
                                <div className="font-semibold">💰 Value Score: {rec.price_score.toFixed(1)}/15</div>
                                <div className="text-sm text-gray-600">Price value within budget</div>
                              </div>
                              <div className="bg-yellow-50 border-l-4 border-yellow-500 p-4 rounded">
                                <div className="font-semibold">⚡ Efficiency Score: {rec.efficiency_score.toFixed(1)}/10</div>
                                <div className="text-sm text-gray-600">Energy efficiency rating</div>
                              </div>
                            </div>
                          </div>

                          {/* AI Reasoning */}
                          <div>
                            <h4 className="font-bold text-gray-900 mb-3">🧠 AI Analysis & Reasoning</h4>
                            <div className="space-y-2">
                              {rec.reasons.map((reason, idx) => (
                                <div key={idx} className="bg-green-50 border-l-4 border-green-500 p-3 rounded text-sm">
                                  ✓ {reason}
                                </div>
                              ))}
                            </div>
                          </div>

                          {/* Complete Specs */}
                          <div>
                            <h4 className="font-bold text-gray-900 mb-3">📋 Complete Specifications</h4>
                            <div className="grid md:grid-cols-3 gap-4 text-sm">
                              <div>
                                <span className="font-semibold">Type:</span> {rec.type}
                              </div>
                              <div>
                                <span className="font-semibold">Segment:</span> {rec.segment}
                              </div>
                              <div>
                                <span className="font-semibold">Predicted Segment:</span> {rec.predicted_segment}
                              </div>
                              <div>
                                <span className="font-semibold">Battery:</span> {rec.battery_kwh} kWh
                              </div>
                              <div>
                                <span className="font-semibold">Range:</span> {rec.range} km
                              </div>
                              <div>
                                <span className="font-semibold">Efficiency:</span> {rec.efficiency.toFixed(1)} km/kWh
                              </div>
                              <div>
                                <span className="font-semibold">Top Speed:</span> {rec.top_speed} km/h
                              </div>
                              <div>
                                <span className="font-semibold">Charging Time:</span> {rec.charging_time} hours
                              </div>
                              <div>
                                <span className="font-semibold">FAME Eligible:</span> {rec.fame_eligible}
                              </div>
                            </div>
                          </div>
                        </div>
                      </details>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {!loading && recommendations.length === 0 && !error && (
              <div className="text-center py-20">
                <Target className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-gray-700 mb-2">No Recommendations Yet</h3>
                <p className="text-gray-500">Set your preferences and click "Get AI Recommendations" to start</p>
              </div>
            )}
          </div>
        </div>

        {/* Info Section */}
        <div className="mt-16 grid md:grid-cols-2 gap-8 max-w-5xl mx-auto">
          <div className="card-gradient p-8">
            <h3 className="text-xl font-bold mb-4 flex items-center">
              <Brain className="w-6 h-6 mr-2 text-blue-600" />
              Machine Learning Model
            </h3>
            <ul className="space-y-2 text-gray-700">
              <li><strong>Algorithm:</strong> Voting Ensemble (RF + GB + RF)</li>
              <li><strong>Accuracy:</strong> 72.06% CV with ±1.8% variance</li>
              <li><strong>Features:</strong> 15 engineered features</li>
              <li><strong>Training Data:</strong> 58 Indian EV models</li>
            </ul>
          </div>

          <div className="card-gradient p-8">
            <h3 className="text-xl font-bold mb-4 flex items-center">
              <Target className="w-6 h-6 mr-2 text-purple-600" />
              Scoring System
            </h3>
            <ul className="space-y-2 text-gray-700">
              <li><strong>🤖 ML Score (50%):</strong> AI confidence</li>
              <li><strong>📏 Range Score (25%):</strong> Daily needs coverage</li>
              <li><strong>💰 Value Score (15%):</strong> Price-performance</li>
              <li><strong>⚡ Efficiency (10%):</strong> Energy rating</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default EVRecommender;
