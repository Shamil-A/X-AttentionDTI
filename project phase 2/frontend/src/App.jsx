import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import Home from './pages/Home';
import Predict from './pages/Predict';
import NetworkBackground from './components/NetworkBackground';
import { Activity } from 'lucide-react';

function App() {
  
  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <Router>
      <div className="min-h-screen flex flex-col text-brand-text font-sans relative">
        
        <NetworkBackground />

        {/* Professional Dark Navigation Bar */}
        <header className="sticky top-0 z-50 bg-brand-bg/90 backdrop-blur-md shadow-lg border-b border-brand-border">
          <div className="max-w-6xl mx-auto px-4 py-4 flex justify-between items-center">
            
            {/* Logo Link (Ensure to="/" is here!) */}
            <Link to="/" onClick={scrollToTop} className="flex items-center space-x-2 transition">
              <Activity size={28} className="text-brand-accent" />
              <span className="text-xl font-bold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-brand-accent to-brand-accent2">
                X-Attention DTI
              </span>
            </Link>
            
            <nav className="space-x-6 flex items-center">
              {/* About Link (Ensure to="/" is here!) */}
              <Link to="/" onClick={scrollToTop} className="text-brand-muted hover:text-brand-accent2 font-medium transition-colors">
                About Project
              </Link>
              
              {/* Predict Link (Ensure to="/predict" is here!) */}
              <Link to="/predict" onClick={scrollToTop} className="bg-gradient-to-r from-brand-accent to-brand-accent2 text-white px-5 py-2 rounded-md font-bold hover:opacity-90 transition-all shadow-md transform hover:scale-105">
                Run Prediction
              </Link>
            </nav>
          </div>
        </header>

        {/* Main Content Area */}
        <main className="flex-grow z-10"> 
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/predict" element={<Predict />} />
          </Routes>
        </main>

        {/* Dark Theme Footer */}
        <footer className="bg-brand-surface border-t border-brand-border text-brand-muted py-6 text-center text-sm z-10"> 
          <p>© {new Date().getFullYear()} X-Attention DTI. Deep Learning for Drug Discovery.</p>
        </footer>
      </div>
    </Router>
  );
}

export default App;