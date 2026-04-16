import React, { useState } from 'react';
import { Upload, Image as ImageIcon, X } from 'lucide-react';

const Dropzone = ({ onImageSelect }) => {
  const [preview, setPreview] = useState(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleFile = (file) => {
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = () => {
        setPreview(reader.result);
        onImageSelect(file, reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const onDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    handleFile(file);
  };

  const onChange = (e) => {
    const file = e.target.files[0];
    handleFile(file);
  };

  const clear = (e) => {
    e.stopPropagation();
    setPreview(null);
    onImageSelect(null, null);
  };

  return (
    <div 
      className={`dropzone ${isDragging ? 'dragging' : ''} ${preview ? 'has-preview' : ''}`}
      onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={onDrop}
      onClick={() => document.getElementById('fileInput').click()}
    >
      <input 
        type="file" 
        id="fileInput" 
        hidden 
        accept="image/*" 
        onChange={onChange}
      />
      
      {preview ? (
        <div className="preview-container">
          <img src={preview} alt="Upload preview" className="preview-image" />
          <button className="clear-btn" onClick={clear}>
            <X size={16} />
          </button>
          <div className="preview-overlay">
            <p>Click to change image</p>
          </div>
        </div>
      ) : (
        <div className="dropzone-content">
          <div className="upload-icon">
            <Upload size={40} />
          </div>
          <h3>Upload Rice Leaf Image</h3>
          <p>Drag and drop or click to browse</p>
          <span className="file-hint">Supports: JPG, PNG, WEBP</span>
        </div>
      )}
    </div>
  );
};

export default Dropzone;
