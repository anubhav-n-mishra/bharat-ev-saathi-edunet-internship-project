import React, { useState, useEffect } from 'react';
import { BarChart3, TrendingUp, PieChart, Calendar, Filter, Download } from 'lucide-react';

interface SalesData {
  month: string;
  brand: string;
  model: string;
  type: string;
  units_sold: number;
  state: string;
}

interface ChartData {
  labels: string[];
  values: number[];
}

const Analytics = () => {
  const [salesData, setSalesData] = useState<SalesData[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedType, setSelectedType] = useState<string>('all');
  const [selectedYear, setSelectedYear] = useState<string>('2023');

  useEffect(() => {
    fetchSalesData();
  }, []);

  const fetchSalesData = async () => {
    try {
      // Generate sample data for visualization
      const sampleData: SalesData[] = [];
      const brands = ['Tata', 'Ather', 'Ola', 'TVS', 'Hero'];
      const types = ['2-Wheeler', '3-Wheeler', '4-Wheeler'];
      const states = ['Maharashtra', 'Delhi', 'Karnataka', 'Gujarat', 'Tamil Nadu'];
      
      for (let month = 1; month <= 12; month++) {
        brands.forEach(brand => {
          types.forEach(type => {
            states.forEach(state => {
              sampleData.push({
                month: `2023-${String(month).padStart(2, '0')}`,
                brand,
                model: `${brand} Model`,
                type,
                units_sold: Math.floor(Math.random() * 2000) + 500,
                state
              });
            });
          });
        });
      }
      
      setSalesData(sampleData);
      setLoading(false);
    } catch (error) {
      console.error('Error fetching sales data:', error);
      setLoading(false);
    }
  };

  const filterData = () => {
    let filtered = salesData.filter(d => d.month.startsWith(selectedYear));
    if (selectedType !== 'all') {
      filtered = filtered.filter(d => d.type === selectedType);
    }
    return filtered;
  };

  const getTopBrands = (): ChartData => {
    const filtered = filterData();
    const brandSales: { [key: string]: number } = {};
    
    filtered.forEach(d => {
      brandSales[d.brand] = (brandSales[d.brand] || 0) + d.units_sold;
    });
    
    const sorted = Object.entries(brandSales)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 5);
    
    return {
      labels: sorted.map(([brand]) => brand),
      values: sorted.map(([, sales]) => sales)
    };
  };

  const getMonthlyTrend = (): ChartData => {
    const filtered = filterData();
    const monthSales: { [key: string]: number } = {};
    
    filtered.forEach(d => {
      monthSales[d.month] = (monthSales[d.month] || 0) + d.units_sold;
    });
    
    const sorted = Object.entries(monthSales).sort(([a], [b]) => a.localeCompare(b));
    
    return {
      labels: sorted.map(([month]) => new Date(month).toLocaleDateString('en-US', { month: 'short' })),
      values: sorted.map(([, sales]) => sales)
    };
  };

  const getTypeDistribution = (): ChartData => {
    const filtered = filterData();
    const typeSales: { [key: string]: number } = {};
    
    filtered.forEach(d => {
      typeSales[d.type] = (typeSales[d.type] || 0) + d.units_sold;
    });
    
    return {
      labels: Object.keys(typeSales),
      values: Object.values(typeSales)
    };
  };

  const getTopStates = (): ChartData => {
    const filtered = filterData();
    const stateSales: { [key: string]: number } = {};
    
    filtered.forEach(d => {
      stateSales[d.state] = (stateSales[d.state] || 0) + d.units_sold;
    });
    
    const sorted = Object.entries(stateSales)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 5);
    
    return {
      labels: sorted.map(([state]) => state),
      values: sorted.map(([, sales]) => sales)
    };
  };

  const calculateStats = () => {
    const filtered = filterData();
    const totalSales = filtered.reduce((sum, d) => sum + d.units_sold, 0);
    const avgMonthlySales = totalSales / 12;
    const topBrand = getTopBrands().labels[0] || 'N/A';
    
    return {
      totalSales,
      avgMonthlySales: Math.round(avgMonthlySales),
      topBrand,
      totalBrands: new Set(filtered.map(d => d.brand)).size
    };
  };

  const renderBarChart = (data: ChartData, title: string, color: string) => {
    const maxValue = Math.max(...data.values);
    
    return (
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h3 className="text-xl font-bold mb-6 flex items-center gap-2">
          <BarChart3 className="text-blue-600" />
          {title}
        </h3>
        <div className="space-y-4">
          {data.labels.map((label, idx) => (
            <div key={idx}>
              <div className="flex justify-between mb-2">
                <span className="text-sm font-semibold text-gray-700">{label}</span>
                <span className="text-sm font-bold text-gray-900">
                  {data.values[idx].toLocaleString()} units
                </span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-3">
                <div
                  className={`${color} h-3 rounded-full transition-all duration-500`}
                  style={{ width: `${(data.values[idx] / maxValue) * 100}%` }}
                ></div>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const renderLineChart = (data: ChartData) => {
    const maxValue = Math.max(...data.values);
    const minValue = Math.min(...data.values);
    const range = maxValue - minValue;
    
    return (
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h3 className="text-xl font-bold mb-6 flex items-center gap-2">
          <TrendingUp className="text-green-600" />
          Monthly Sales Trend
        </h3>
        <div className="h-64 flex items-end justify-between gap-2">
          {data.values.map((value, idx) => {
            const height = ((value - minValue) / range) * 100;
            return (
              <div key={idx} className="flex-1 flex flex-col items-center">
                <div
                  className="w-full bg-linear-to-t from-green-500 to-green-300 rounded-t-lg transition-all duration-500 hover:from-green-600 hover:to-green-400 cursor-pointer relative group"
                  style={{ height: `${height}%`, minHeight: '20px' }}
                >
                  <div className="absolute -top-8 left-1/2 transform -translate-x-1/2 bg-gray-800 text-white text-xs px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">
                    {value.toLocaleString()}
                  </div>
                </div>
                <span className="text-xs mt-2 text-gray-600">{data.labels[idx]}</span>
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  const renderPieChart = (data: ChartData) => {
    const total = data.values.reduce((sum, v) => sum + v, 0);
    const colors = ['bg-blue-500', 'bg-green-500', 'bg-purple-500', 'bg-orange-500', 'bg-pink-500'];
    
    return (
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h3 className="text-xl font-bold mb-6 flex items-center gap-2">
          <PieChart className="text-purple-600" />
          Vehicle Type Distribution
        </h3>
        <div className="space-y-4">
          {data.labels.map((label, idx) => {
            const percentage = ((data.values[idx] / total) * 100).toFixed(1);
            return (
              <div key={idx}>
                <div className="flex justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <div className={`w-4 h-4 ${colors[idx % colors.length]} rounded`}></div>
                    <span className="text-sm font-semibold text-gray-700">{label}</span>
                  </div>
                  <span className="text-sm font-bold text-gray-900">{percentage}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className={`${colors[idx % colors.length]} h-2 rounded-full`}
                    style={{ width: `${percentage}%` }}
                  ></div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <BarChart3 className="w-16 h-16 text-blue-600 mx-auto mb-4 animate-pulse" />
          <p className="text-xl text-gray-600">Loading analytics...</p>
        </div>
      </div>
    );
  }

  const stats = calculateStats();
  const topBrands = getTopBrands();
  const monthlyTrend = getMonthlyTrend();
  const typeDistribution = getTypeDistribution();
  const topStates = getTopStates();

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="container mx-auto px-4">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-4 flex items-center gap-3">
            <BarChart3 className="text-blue-600" size={40} />
            EV Sales Analytics
          </h1>
          <p className="text-gray-600 text-lg">
            Comprehensive analysis of electric vehicle sales across India
          </p>
        </div>

        {/* Filters */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
          <div className="flex flex-wrap gap-4 items-center">
            <div className="flex items-center gap-2">
              <Filter className="text-gray-600" size={20} />
              <span className="font-semibold text-gray-700">Filters:</span>
            </div>
            
            <div>
              <label className="block text-sm text-gray-600 mb-1">Year</label>
              <select
                value={selectedYear}
                onChange={(e) => setSelectedYear(e.target.value)}
                className="px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="2023">2023</option>
                <option value="2024">2024</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm text-gray-600 mb-1">Vehicle Type</label>
              <select
                value={selectedType}
                onChange={(e) => setSelectedType(e.target.value)}
                className="px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="all">All Types</option>
                <option value="2-Wheeler">2-Wheeler</option>
                <option value="3-Wheeler">3-Wheeler</option>
                <option value="4-Wheeler">4-Wheeler</option>
              </select>
            </div>
            
            <button className="ml-auto px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center gap-2">
              <Download size={18} />
              Export Report
            </button>
          </div>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="bg-linear-to-br from-blue-500 to-blue-600 rounded-xl p-6 text-white shadow-lg">
            <div className="flex items-center justify-between mb-2">
              <TrendingUp size={32} />
              <Calendar size={24} className="opacity-80" />
            </div>
            <p className="text-blue-100 text-sm mb-1">Total Sales ({selectedYear})</p>
            <p className="text-3xl font-bold">{stats.totalSales.toLocaleString()}</p>
            <p className="text-blue-100 text-xs mt-2">units sold</p>
          </div>

          <div className="bg-linear-to-br from-green-500 to-green-600 rounded-xl p-6 text-white shadow-lg">
            <div className="flex items-center justify-between mb-2">
              <BarChart3 size={32} />
            </div>
            <p className="text-green-100 text-sm mb-1">Avg. Monthly Sales</p>
            <p className="text-3xl font-bold">{stats.avgMonthlySales.toLocaleString()}</p>
            <p className="text-green-100 text-xs mt-2">units per month</p>
          </div>

          <div className="bg-linear-to-br from-purple-500 to-purple-600 rounded-xl p-6 text-white shadow-lg">
            <div className="flex items-center justify-between mb-2">
              <PieChart size={32} />
            </div>
            <p className="text-purple-100 text-sm mb-1">Top Brand</p>
            <p className="text-3xl font-bold">{stats.topBrand}</p>
            <p className="text-purple-100 text-xs mt-2">market leader</p>
          </div>

          <div className="bg-linear-to-br from-orange-500 to-orange-600 rounded-xl p-6 text-white shadow-lg">
            <div className="flex items-center justify-between mb-2">
              <BarChart3 size={32} />
            </div>
            <p className="text-orange-100 text-sm mb-1">Active Brands</p>
            <p className="text-3xl font-bold">{stats.totalBrands}</p>
            <p className="text-orange-100 text-xs mt-2">in market</p>
          </div>
        </div>

        {/* Charts Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {renderLineChart(monthlyTrend)}
          {renderBarChart(topBrands, 'Top 5 Brands by Sales', 'bg-blue-500')}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {renderPieChart(typeDistribution)}
          {renderBarChart(topStates, 'Top 5 States by Sales', 'bg-green-500')}
        </div>

        {/* Insights */}
        <div className="mt-8 bg-linear-to-r from-blue-50 to-purple-50 rounded-xl p-6 border border-blue-200">
          <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
            <TrendingUp className="text-blue-600" />
            Key Insights
          </h3>
          <ul className="space-y-2 text-gray-700">
            <li className="flex items-start gap-2">
              <span className="text-blue-600 font-bold">•</span>
              <span><strong>{topBrands.labels[0]}</strong> leads the market with {topBrands.values[0].toLocaleString()} units sold in {selectedYear}</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-green-600 font-bold">•</span>
              <span>{typeDistribution.labels[0]} segment shows strongest performance with {((typeDistribution.values[0] / typeDistribution.values.reduce((a, b) => a + b, 0)) * 100).toFixed(1)}% market share</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-purple-600 font-bold">•</span>
              <span><strong>{topStates.labels[0]}</strong> is the top-performing state with {topStates.values[0].toLocaleString()} total sales</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-orange-600 font-bold">•</span>
              <span>Average monthly growth indicates {stats.avgMonthlySales > 50000 ? 'strong' : 'steady'} adoption of electric vehicles</span>
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default Analytics;
