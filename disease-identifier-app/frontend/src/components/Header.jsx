import React from 'react';
import { Leaf } from 'lucide-react';

const Header = () => {
  return (
    <header className="header">
      <div className="logo-container">
        <div className="logo-icon">
          <Leaf size={32} color="#10b981" />
        </div>
        <div className="logo-text">
          <h1>RiceLeaf<span>AI</span></h1>
          <p>Dual-Stage Disease Detection</p>
        </div>
      </div>
    </header>
  );
};

export default Header;
