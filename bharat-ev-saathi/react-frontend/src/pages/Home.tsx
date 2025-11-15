import { Link } from 'react-router-dom';
import { Zap, Target, Brain, TrendingUp, Shield, Users } from 'lucide-react';

const Home = () => {
  const stats = [
    { label: 'EV Models', value: '58', icon: Zap },
    { label: 'Charging Stations', value: '458', icon: TrendingUp },
    { label: 'Cities Covered', value: '15+', icon: Users },
    { label: 'ML Accuracy', value: '72%', icon: Brain },
  ];

  const features = [
    {
      icon: Brain,
      title: 'AI-Powered Recommendations',
      description: 'Get personalized EV suggestions using our production ML model with 72% accuracy.',
      gradient: 'from-blue-500 to-cyan-500',
    },
    {
      icon: Target,
      title: 'Smart Matching',
      description: 'Match EVs based on your budget, daily usage, and preferences with intelligent algorithms.',
      gradient: 'from-purple-500 to-pink-500',
    },
    {
      icon: Shield,
      title: 'FAME Subsidy Calculator',
      description: 'Know your exact savings with instant subsidy calculations for eligible vehicles.',
      gradient: 'from-green-500 to-emerald-500',
    },
    {
      icon: TrendingUp,
      title: 'Market Analytics',
      description: 'Explore comprehensive EV market insights, trends, and adoption patterns across India.',
      gradient: 'from-orange-500 to-red-500',
    },
  ];

  return (
    <div className="animate-fade-in">
      {/* Hero Section */}
      <section className="bg-linear-to-br from-blue-600 to-purple-600 text-white py-12 md:py-20 px-4">
        <div className="container mx-auto">
          <div className="max-w-4xl mx-auto text-center">
            <h1 className="text-4xl sm:text-5xl md:text-6xl font-bold mb-4 md:mb-6">
              Bharat EV Saathi
            </h1>
            <p className="text-xl sm:text-2xl mb-3 md:mb-4 opacity-95">भारत EV साथी</p>
            <p className="text-lg sm:text-xl mb-6 md:mb-8 leading-relaxed opacity-90">
              Your Intelligent Companion for Electric Vehicle Decisions in India
            </p>
            <p className="text-base sm:text-lg mb-8 md:mb-10 opacity-85">
              Powered by Machine Learning • 72% Prediction Accuracy • 58+ EV Models
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
              <Link to="/recommender" className="flex items-center gap-2 bg-white text-blue-600 font-semibold px-6 md:px-8 py-3 md:py-4 rounded-lg hover:bg-gray-100 transition-all transform hover:scale-105 shadow-lg w-full sm:w-auto justify-center">
                <Target className="w-5 h-5" />
                <span>Get EV Recommendations</span>
              </Link>
              <Link to="/analytics" className="btn-secondary inline-flex items-center justify-center space-x-2">
                <TrendingUp className="w-5 h-5" />
                <span>View Analytics</span>
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-12 md:py-16 bg-white">
        <div className="container mx-auto px-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 md:gap-6">
            {stats.map((stat, index) => {
              const Icon = stat.icon;
              return (
                <div key={index} className="card p-4 md:p-6 text-center hover:scale-105 transition-transform">
                  <div className="bg-linear-to-br from-blue-100 to-purple-100 w-10 h-10 md:w-12 md:h-12 rounded-full flex items-center justify-center mx-auto mb-2 md:mb-3">
                    <Icon className="w-5 h-5 md:w-6 md:h-6 text-blue-600" />
                  </div>
                  <div className="text-2xl md:text-3xl font-bold text-gray-900 mb-1">{stat.value}</div>
                  <div className="text-xs md:text-sm text-gray-600">{stat.label}</div>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-12 md:py-20 bg-gray-50">
        <div className="container mx-auto px-4">
          <div className="text-center mb-8 md:mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-3 md:mb-4">Why Choose Bharat EV Saathi?</h2>
            <p className="text-lg md:text-xl text-gray-600">Comprehensive, data-driven EV decision support</p>
          </div>
          
          <div className="grid sm:grid-cols-2 gap-6 md:gap-8 max-w-5xl mx-auto">
            {features.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <div key={index} className="card p-6 md:p-8 hover:scale-105 transition-transform animate-slide-up" 
                     style={{animationDelay: `${index * 100}ms`}}>
                  <div className={`bg-linear-to-r ${feature.gradient} w-12 h-12 md:w-14 md:h-14 rounded-xl flex items-center justify-center mb-3 md:mb-4`}>
                    <Icon className="w-6 h-6 md:w-7 md:h-7 text-white" />
                  </div>
                  <h3 className="text-lg md:text-xl font-bold text-gray-900 mb-2 md:mb-3">{feature.title}</h3>
                  <p className="text-sm md:text-base text-gray-600 leading-relaxed">{feature.description}</p>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-12 md:py-16 bg-linear-to-r from-blue-600 to-purple-600">
        <div className="container mx-auto px-4 text-center">
          <h2 className="text-2xl sm:text-3xl md:text-4xl font-bold text-white mb-3 md:mb-4">
            Ready to Find Your Perfect EV?
          </h2>
          <p className="text-lg md:text-xl text-blue-100 mb-6 md:mb-8">
            Get AI-powered recommendations tailored to your needs in minutes
          </p>
          <Link to="/recommender" className="inline-flex items-center space-x-2 bg-white text-blue-600 px-6 md:px-8 py-3 md:py-4 rounded-lg font-bold text-base md:text-lg hover:bg-gray-100 transition shadow-xl">
            <Target className="w-6 h-6" />
            <span>Start Recommendation Engine</span>
          </Link>
        </div>
      </section>
    </div>
  );
};

export default Home;
