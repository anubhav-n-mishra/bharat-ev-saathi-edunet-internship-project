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
      <section className="bg-gradient-to-br from-blue-50 via-white to-purple-50 py-20">
        <div className="container mx-auto px-4">
          <div className="max-w-4xl mx-auto text-center">
            <h1 className="text-5xl md:text-6xl font-bold mb-6">
              <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                Bharat EV Saathi
              </span>
            </h1>
            <p className="text-2xl text-gray-600 mb-4">भारत EV साथी</p>
            <p className="text-xl text-gray-700 mb-8 leading-relaxed">
              Your Intelligent Companion for Electric Vehicle Decisions in India
            </p>
            <p className="text-lg text-gray-600 mb-10">
              Powered by Machine Learning • 72% Prediction Accuracy • 58+ EV Models
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link to="/recommender" className="btn-primary inline-flex items-center justify-center space-x-2">
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
      <section className="py-12 bg-white">
        <div className="container mx-auto px-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            {stats.map((stat, index) => {
              const Icon = stat.icon;
              return (
                <div key={index} className="card p-6 text-center hover:scale-105 transition-transform">
                  <div className="bg-gradient-to-br from-blue-100 to-purple-100 w-12 h-12 rounded-full flex items-center justify-center mx-auto mb-3">
                    <Icon className="w-6 h-6 text-blue-600" />
                  </div>
                  <div className="text-3xl font-bold text-gray-900 mb-1">{stat.value}</div>
                  <div className="text-sm text-gray-600">{stat.label}</div>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 bg-gray-50">
        <div className="container mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-gray-900 mb-4">Why Choose Bharat EV Saathi?</h2>
            <p className="text-xl text-gray-600">Comprehensive, data-driven EV decision support</p>
          </div>
          
          <div className="grid md:grid-cols-2 gap-8 max-w-5xl mx-auto">
            {features.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <div key={index} className="card p-8 hover:scale-105 transition-transform animate-slide-up" 
                     style={{animationDelay: `${index * 100}ms`}}>
                  <div className={`bg-gradient-to-r ${feature.gradient} w-14 h-14 rounded-xl flex items-center justify-center mb-4`}>
                    <Icon className="w-7 h-7 text-white" />
                  </div>
                  <h3 className="text-xl font-bold text-gray-900 mb-3">{feature.title}</h3>
                  <p className="text-gray-600 leading-relaxed">{feature.description}</p>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-16 bg-gradient-to-r from-blue-600 to-purple-600">
        <div className="container mx-auto px-4 text-center">
          <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
            Ready to Find Your Perfect EV?
          </h2>
          <p className="text-xl text-blue-100 mb-8">
            Get AI-powered recommendations tailored to your needs in minutes
          </p>
          <Link to="/recommender" className="inline-flex items-center space-x-2 bg-white text-blue-600 px-8 py-4 rounded-lg font-bold text-lg hover:bg-gray-100 transition shadow-xl">
            <Target className="w-6 h-6" />
            <span>Start Recommendation Engine</span>
          </Link>
        </div>
      </section>
    </div>
  );
};

export default Home;
