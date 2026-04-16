import React, { useRef, useState } from "react";
import "./ImageUploader.css";

/**
 * ImageUploader
 * -------------
 * Drag-and-drop / click-to-browse file picker.
 *
 * Props
 * -----
 * onImageSelect(file: File) – called when the user picks a valid image.
 */
function ImageUploader({ onImageSelect }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  const handleFile = (file) => {
    if (!file || !file.type.startsWith("image/")) return;
    onImageSelect(file);
  };

  /* ---- drag events ---- */
  const onDragOver = (e) => { e.preventDefault(); setDragging(true); };
  const onDragLeave = () => setDragging(false);
  const onDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    handleFile(e.dataTransfer.files[0]);
  };

  /* ---- click-to-browse ---- */
  const onInputChange = (e) => handleFile(e.target.files[0]);

  return (
    <div
      className={`uploader${dragging ? " uploader--dragging" : ""}`}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
      onClick={() => inputRef.current.click()}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => e.key === "Enter" && inputRef.current.click()}
      aria-label="Upload a rice leaf image"
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="uploader__input"
        onChange={onInputChange}
      />

      <div className="uploader__icon" aria-hidden="true">🌿</div>
      <p className="uploader__primary">Drop a rice-leaf image here</p>
      <p className="uploader__secondary">or click to browse</p>
      <p className="uploader__hint">Supported: JPG, PNG, WEBP</p>
    </div>
  );
}

export default ImageUploader;
