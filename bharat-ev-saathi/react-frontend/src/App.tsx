import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import Footer from './components/Footer';
import Home from './pages/Home';
import EVRecommender from './pages/EVRecommender';
import Analytics from './pages/Analytics';
import Chatbot from './pages/Chatbot';
import ChargingStations from './pages/ChargingStations';
import SubsidyCalculator from './pages/SubsidyCalculator';

function App() {
  return (
    <Router>
      <div className="flex flex-col min-h-screen">
        <Header />
        <main className="flex-grow">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/recommender" element={<EVRecommender />} />
            <Route path="/subsidy" element={<SubsidyCalculator />} />
            <Route path="/analytics" element={<Analytics />} />
            <Route path="/chatbot" element={<Chatbot />} />
            <Route path="/charging-stations" element={<ChargingStations />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </Router>
  );
}

export default App;
