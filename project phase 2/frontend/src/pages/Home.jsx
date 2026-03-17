import { ArrowRight, Dna, Network, Zap } from 'lucide-react';
import { Link } from 'react-router-dom';

export default function Home() {
  return (
    <div className="max-w-6xl mx-auto px-4 py-12">
      <section className="text-center py-16 lg:py-24">
        {/* Main Title with Explainer Gradient */}
        <h1 className="text-4xl md:text-6xl font-extrabold text-[#e8eaf0] mb-6 tracking-tight">
          Next-Generation <span className="text-transparent bg-clip-text bg-gradient-to-r from-[#7c6cf4] to-[#38d9f5]">Drug-Target Interaction</span>
        </h1>
        {/* Subtitle with readable dark-mode gray */}
        <p className="text-xl text-[#8b92a5] mb-10 max-w-3xl mx-auto leading-relaxed">
          X-Attention DTI leverages advanced hypergraph neural networks and protein language models (ESM-2) to predict binding affinities with unprecedented accuracy.
        </p>
        {/* Gradient Button matching the Predict page */}
        <Link to="/predict" className="inline-flex items-center space-x-2 bg-gradient-to-r from-[#7c6cf4] to-[#38d9f5] text-white px-8 py-4 rounded-lg text-lg font-bold hover:opacity-90 transition-all shadow-lg transform hover:scale-105">
          <span>Start Predicting</span>
          <ArrowRight size={20} />
        </Link>
      </section>

      {/* Dark mode divider line */}
      <hr className="border-[#1f2340] my-12" />

      <section className="py-12">
        <h2 className="text-3xl font-bold text-center mb-12 text-[#e8eaf0]">Under the Hood</h2>
        <div className="grid md:grid-cols-3 gap-8">
          
          {/* Card 1: Drug Encoder (Midnight Blue Background) */}
          <div className="bg-[#111320] p-8 rounded-xl shadow-xl border border-[#1f2340] hover:border-[#7c6cf4]/50 transition-colors">
            {/* Purple Icon Container */}
            <div className="w-12 h-12 bg-[#1a1e35] text-[#7c6cf4] rounded-lg flex items-center justify-center mb-6">
              <Network size={28} />
            </div>
            <h3 className="text-xl font-bold mb-3 text-[#e8eaf0]">Hypergraph Drug Encoder</h3>
            <p className="text-[#8b92a5] leading-relaxed">
              Captures complex multi-way atomic interactions and ring structures using degree-normalized hypergraph convolutions.
            </p>
          </div>

          {/* Card 2: Protein Modeling */}
          <div className="bg-[#111320] p-8 rounded-xl shadow-xl border border-[#1f2340] hover:border-[#38d9f5]/50 transition-colors">
            {/* Cyan Icon Container */}
            <div className="w-12 h-12 bg-[#1a1e35] text-[#38d9f5] rounded-lg flex items-center justify-center mb-6">
              <Dna size={28} />
            </div>
            <h3 className="text-xl font-bold mb-3 text-[#e8eaf0]">ESM-2 Protein Modeling</h3>
            <p className="text-[#8b92a5] leading-relaxed">
              Utilizes Facebook's state-of-the-art evolutionary scale modeling. We fuse representations from student and teacher models.
            </p>
          </div>

          {/* Card 3: Cross-Attention */}
          <div className="bg-[#111320] p-8 rounded-xl shadow-xl border border-[#1f2340] hover:border-[#f97316]/50 transition-colors">
            {/* Orange Icon Container */}
            <div className="w-12 h-12 bg-[#1a1e35] text-[#f97316] rounded-lg flex items-center justify-center mb-6">
              <Zap size={28} />
            </div>
            <h3 className="text-xl font-bold mb-3 text-[#e8eaf0]">Cross-Attention</h3>
            <p className="text-[#8b92a5] leading-relaxed">
              Dynamically aligns the sequence-level protein embeddings with the hypergraph node-level drug embeddings.
            </p>
          </div>

        </div>
      </section>
    </div>
  );
}