import { Github, Linkedin, Mail } from 'lucide-react';

const Footer = () => {
  return (
    <footer className="bg-gray-900 text-white mt-auto">
      <div className="container mx-auto px-4 py-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* About */}
          <div>
            <h3 className="text-lg font-bold mb-4">Bharat EV Saathi</h3>
            <p className="text-gray-400 text-sm">
              Your intelligent companion for electric vehicle decisions in India. 
              Powered by AI and ML for personalized recommendations.
            </p>
          </div>
          
          {/* Quick Links */}
          <div>
            <h3 className="text-lg font-bold mb-4">Quick Links</h3>
            <ul className="space-y-2 text-sm">
              <li><a href="/" className="text-gray-400 hover:text-white transition">Home</a></li>
              <li><a href="/recommender" className="text-gray-400 hover:text-white transition">EV Recommender</a></li>
              <li><a href="/analytics" className="text-gray-400 hover:text-white transition">Analytics</a></li>
              <li><a href="/chatbot" className="text-gray-400 hover:text-white transition">Chatbot</a></li>
            </ul>
          </div>
          
          {/* Connect */}
          <div>
            <h3 className="text-lg font-bold mb-4">Connect</h3>
            <div className="flex space-x-4">
              <a href="https://github.com/anubhav-n-mishra" target="_blank" rel="noopener noreferrer" 
                 className="bg-gray-800 p-2 rounded-lg hover:bg-gray-700 transition">
                <Github className="w-5 h-5" />
              </a>
              <a href="https://linkedin.com" target="_blank" rel="noopener noreferrer" 
                 className="bg-gray-800 p-2 rounded-lg hover:bg-gray-700 transition">
                <Linkedin className="w-5 h-5" />
              </a>
              <a href="mailto:contact@bharatev.com" 
                 className="bg-gray-800 p-2 rounded-lg hover:bg-gray-700 transition">
                <Mail className="w-5 h-5" />
              </a>
            </div>
            <p className="text-gray-400 text-sm mt-4">
              © 2025 Bharat EV Saathi<br/>
              Powered by Production ML Model (72% Accuracy)
            </p>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
