import { Link, useLocation } from 'react-router-dom';
import { Zap, Home, Target, BarChart3, MessageSquare, MapPin, Calculator } from 'lucide-react';

const Header = () => {
  const location = useLocation();
  
  const isActive = (path: string) => location.pathname === path;
  
  return (
    <header className="bg-white shadow-md sticky top-0 z-50">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link to="/" className="flex items-center space-x-2 group">
            <div className="bg-gradient-to-r from-blue-600 to-purple-600 p-2 rounded-lg group-hover:scale-110 transition-transform">
              <Zap className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                Bharat EV Saathi
              </h1>
              <p className="text-xs text-gray-500">भारत EV साथी</p>
            </div>
          </Link>
          
          {/* Navigation */}
          <nav className="hidden md:flex space-x-1">
            <Link
              to="/"
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all ${
                isActive('/') 
                  ? 'bg-blue-100 text-blue-600 font-semibold' 
                  : 'text-gray-600 hover:bg-gray-100'
              }`}
            >
              <Home className="w-4 h-4" />
              <span>Home</span>
            </Link>
            
            <Link
              to="/recommender"
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all ${
                isActive('/recommender') 
                  ? 'bg-blue-100 text-blue-600 font-semibold' 
                  : 'text-gray-600 hover:bg-gray-100'
              }`}
            >
              <Target className="w-4 h-4" />
              <span>EV Recommender</span>
            </Link>
            
            <Link
              to="/subsidy"
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all ${
                isActive('/subsidy') 
                  ? 'bg-blue-100 text-blue-600 font-semibold' 
                  : 'text-gray-600 hover:bg-gray-100'
              }`}
            >
              <Calculator className="w-4 h-4" />
              <span>Subsidy Calculator</span>
            </Link>
            
            <Link
              to="/analytics"
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all ${
                isActive('/analytics') 
                  ? 'bg-blue-100 text-blue-600 font-semibold' 
                  : 'text-gray-600 hover:bg-gray-100'
              }`}
            >
              <BarChart3 className="w-4 h-4" />
              <span>Analytics</span>
            </Link>
            
            <Link
              to="/chatbot"
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all ${
                isActive('/chatbot') 
                  ? 'bg-blue-100 text-blue-600 font-semibold' 
                  : 'text-gray-600 hover:bg-gray-100'
              }`}
            >
              <MessageSquare className="w-4 h-4" />
              <span>Chatbot</span>
            </Link>
            
            <Link
              to="/charging-stations"
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all ${
                isActive('/charging-stations') 
                  ? 'bg-blue-100 text-blue-600 font-semibold' 
                  : 'text-gray-600 hover:bg-gray-100'
              }`}
            >
              <MapPin className="w-4 h-4" />
              <span>Charging Stations</span>
            </Link>
          </nav>
          
          {/* CTA Button */}
          <button className="btn-primary text-sm py-2 px-4 hidden lg:block">
            Get Started
          </button>
        </div>
      </div>
    </header>
  );
};

export default Header;
