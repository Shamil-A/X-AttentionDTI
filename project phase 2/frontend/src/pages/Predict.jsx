import { useState } from 'react';
import axios from 'axios';
import { Activity, AlertCircle, CheckCircle2, RotateCcw, FlaskConical, ExternalLink } from 'lucide-react';

export default function Predict() {
  const [smiles, setSmiles] = useState('');
  const [protein, setProtein] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await axios.post('https://shamil777-x-attentiondti.hf.space/api/predict', {
        smiles,
        protein
      });
      
      setResult(response.data.prediction);
    } catch (err) {
      setError(err.response?.data?.error || 'An error occurred during prediction.');
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setSmiles('');
    setProtein('');
    setResult(null);
    setError(null);
  };

  const handleLoadExample = () => {
    setSmiles('CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5');
    setProtein('MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR');
    setResult(null);
    setError(null);
  };

  return (
    <div className="max-w-4xl mx-auto px-4 py-12 flex flex-col items-center">
      
      <div className="w-full bg-[#111320] rounded-2xl shadow-2xl border border-[#1f2340] p-8">
        
        <div className="flex flex-col md:flex-row md:items-center justify-between mb-6 border-b border-[#1f2340] pb-4">
          <h2 className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-[#7c6cf4] to-[#38d9f5] flex items-center mb-4 md:mb-0">
            <Activity className="mr-3 text-[#7c6cf4]" />
            Run DTI Prediction
          </h2>
          <button
            type="button"
            onClick={handleLoadExample}
            className="flex items-center text-sm font-medium text-[#38d9f5] bg-[#1a1e35] hover:bg-[#1f2340] px-4 py-2 rounded-lg transition-colors border border-[#1f2340]"
          >
            <FlaskConical size={16} className="mr-2" />
            Load Example Data
          </button>
        </div>
        
        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label className="block text-sm font-semibold text-[#e8eaf0] mb-2">
              Drug (SMILES String)
            </label>
            <textarea
              value={smiles}
              onChange={(e) => setSmiles(e.target.value)}
              placeholder="e.g., CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5"
              className="w-full p-4 bg-[#08090f] border border-[#1f2340] text-[#e8eaf0] placeholder-[#6b7280] rounded-lg focus:outline-none focus:ring-2 focus:ring-[#7c6cf4] focus:border-[#7c6cf4] font-mono text-sm shadow-sm transition-all"
              rows="3"
              required
            />
          </div>

          <div>
            <label className="block text-sm font-semibold text-[#e8eaf0] mb-2">
              Target Protein (Amino Acid Sequence)
            </label>
            <textarea
              value={protein}
              onChange={(e) => setProtein(e.target.value)}
              placeholder="e.g., MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR"
              className="w-full p-4 bg-[#08090f] border border-[#1f2340] text-[#e8eaf0] placeholder-[#6b7280] rounded-lg focus:outline-none focus:ring-2 focus:ring-[#7c6cf4] focus:border-[#7c6cf4] font-mono text-sm shadow-sm transition-all"
              rows="6"
              required
            />
          </div>

          <div className="flex space-x-4">
            <button
              type="button"
              onClick={handleClear}
              disabled={loading || (!smiles && !protein && !result)}
              className="w-1/4 bg-[#1a1e35] text-[#e8eaf0] border border-[#1f2340] font-bold py-4 rounded-lg hover:bg-[#1f2340] transition disabled:opacity-50 flex justify-center items-center shadow-sm"
            >
              <RotateCcw className="mr-2" size={20} />
              Clear
            </button>

            <button
              type="submit"
              disabled={loading || !smiles || !protein}
              className="w-3/4 bg-gradient-to-r from-[#7c6cf4] to-[#38d9f5] text-white font-bold py-4 rounded-lg hover:opacity-90 transition-all disabled:opacity-50 flex justify-center items-center shadow-lg transform active:scale-[0.98]"
            >
              {loading ? 'Analyzing Binding Affinity...' : 'Predict Affinity (KIBA)'}
            </button>
          </div>
        </form>

        {error && (
          <div className="mt-8 p-4 bg-red-900/20 border border-red-500/50 text-red-400 rounded-lg flex items-start animate-fade-in">
            <AlertCircle className="mr-3 shrink-0 mt-0.5" size={20} />
            <p>{error}</p>
          </div>
        )}

        {/* --- UPDATED RESULT BOX --- */}
        {result !== null && (
          <div className="mt-8 p-8 bg-[#08090f] border border-[#38d9f5]/30 rounded-xl flex flex-col items-center justify-center text-center shadow-[0_0_20px_rgba(56,217,245,0.1)] animate-fade-in">
            
            {/* Glowing Icon Container */}
            <div className="w-16 h-16 bg-[#111320] rounded-full flex items-center justify-center mb-4 border border-[#38d9f5]/40 shadow-[0_0_15px_rgba(56,217,245,0.2)]">
              <CheckCircle2 className="text-[#38d9f5]" size={32} />
            </div>
            
            <h3 className="text-xl font-bold text-[#e8eaf0] mb-2 tracking-wide">Predicted Binding Affinity</h3>
            
            {/* Massive, Bright Gradient Number */}
            <p className="text-6xl font-black text-transparent bg-clip-text bg-gradient-to-r from-[#38d9f5] to-[#7c6cf4] mt-2 drop-shadow-md">
              {result}
            </p>
            
          </div>
        )}
      </div>

      <div className="mt-8">
        <a 
          href="/pipeline_explainer.html" 
          target="_blank" 
          rel="noopener noreferrer"
          className="flex items-center px-6 py-3 rounded-full border border-[#7c6cf4]/30 bg-[#1a1e35] text-[#38d9f5] hover:text-white hover:border-[#7c6cf4] hover:bg-[#111320] transition-all shadow-lg font-medium"
        >
          <span className="mr-2">🧬</span> 
          View Interactive Pipeline Explainer
          <ExternalLink size={16} className="ml-2" />
        </a>
      </div>
    </div>
  );
}