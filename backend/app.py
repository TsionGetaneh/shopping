from flask import Flask, request, send_file, render_template_string, jsonify
from inference_test import generate_tryon
import os
import glob

app = Flask(__name__)

if not os.path.exists("uploads"):
    os.makedirs("uploads")

# Cloth gallery directory
CLOTH_GALLERY_DIR = "datasets/cloth"
if not os.path.exists(CLOTH_GALLERY_DIR):
    os.makedirs(CLOTH_GALLERY_DIR)

INDEX_HTML = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Virtual Try-On Studio</title>
    <style>
      * { margin: 0; padding: 0; box-sizing: border-box; }
      body { 
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
        padding: 20px;
      }
      .container {
        max-width: 1400px;
        margin: 0 auto;
        background: white;
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        overflow: hidden;
      }
      .header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        text-align: center;
      }
      .header h1 { font-size: 2.5em; margin-bottom: 10px; }
      .header p { font-size: 1.1em; opacity: 0.9; }
      
      .dashboard {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 30px;
        padding: 30px;
      }
      
      .section {
        background: #f8f9fa;
        border-radius: 15px;
        padding: 25px;
        border: 2px solid #e9ecef;
      }
      
      .section h2 {
        color: #667eea;
        margin-bottom: 20px;
        font-size: 1.5em;
        border-bottom: 2px solid #667eea;
        padding-bottom: 10px;
      }
      
      .upload-area {
        border: 3px dashed #667eea;
        border-radius: 10px;
        padding: 30px;
        text-align: center;
        background: white;
        cursor: pointer;
        transition: all 0.3s;
        margin-bottom: 15px;
      }
      .upload-area:hover {
        border-color: #764ba2;
        background: #f8f9ff;
      }
      .upload-area.dragover {
        border-color: #764ba2;
        background: #e8eaff;
      }
      
      input[type="file"] {
        display: none;
      }
      
      .preview-img {
        max-width: 100%;
        max-height: 300px;
        border-radius: 10px;
        margin-top: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
      }
      
      .cloth-gallery {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
        gap: 15px;
        max-height: 400px;
        overflow-y: auto;
        padding: 10px;
      }
      
      .cloth-item {
        position: relative;
        border: 3px solid transparent;
        border-radius: 10px;
        overflow: hidden;
        cursor: pointer;
        transition: all 0.3s;
        background: white;
      }
      .cloth-item:hover {
        transform: scale(1.05);
        border-color: #667eea;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
      }
      .cloth-item.selected {
        border-color: #764ba2;
        box-shadow: 0 0 0 3px rgba(118, 75, 162, 0.3);
      }
      
      .cloth-item img {
        width: 100%;
        height: 150px;
        object-fit: cover;
      }
      
      .btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 15px 40px;
        font-size: 1.1em;
        border-radius: 50px;
        cursor: pointer;
        transition: all 0.3s;
        width: 100%;
        margin-top: 20px;
        font-weight: bold;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
      }
      .btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.6);
      }
      .btn:active {
        transform: translateY(0);
      }
      .btn:disabled {
        opacity: 0.6;
        cursor: not-allowed;
      }
      
      .result-section {
        grid-column: 1 / -1;
        text-align: center;
        padding: 30px;
      }
      
      .result-img {
        max-width: 100%;
        max-height: 600px;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        margin-top: 20px;
      }
      
      .loading {
        display: none;
        text-align: center;
        padding: 20px;
      }
      .loading.active {
        display: block;
      }
      .spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #667eea;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        animation: spin 1s linear infinite;
        margin: 0 auto;
      }
      @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
      }
      
      .status {
        padding: 15px;
        border-radius: 10px;
        margin: 15px 0;
        display: none;
      }
      .status.success {
        background: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
        display: block;
      }
      .status.error {
        background: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
        display: block;
      }
    </style>
  </head>
  <body>
    <div class="container">
      <div class="header">
        <h1>👗 Virtual Try-On Studio</h1>
        <p>Upload a person image and select clothing to see them try it on!</p>
      </div>
      
      <div class="dashboard">
        <div class="section">
          <h2>📸 Person Image</h2>
          <div class="upload-area" id="person-upload">
            <p>Click or drag to upload person image</p>
            <input type="file" id="person-file" accept="image/*">
            <div id="person-preview"></div>
          </div>
        </div>
        
        <div class="section">
          <h2>👕 Clothing Selection</h2>
          <div class="upload-area" id="cloth-upload">
            <p>Click or drag to upload cloth image</p>
            <input type="file" id="cloth-file" accept="image/*">
            <div id="cloth-preview"></div>
          </div>
          
          <h3 style="margin-top: 20px; color: #667eea;">Or select from gallery:</h3>
          <div class="cloth-gallery" id="cloth-gallery">
            <!-- Cloth items will be loaded here -->
          </div>
        </div>
        
        <div class="result-section">
          <button class="btn" id="tryon-btn" disabled>✨ Generate Try-On</button>
          <div class="loading" id="loading">
            <div class="spinner"></div>
            <p style="margin-top: 15px;">Processing... This may take a moment</p>
          </div>
          <div class="status" id="status"></div>
          <div id="result-container">
            <h2 style="color: #667eea; margin-top: 30px;">Result will appear here</h2>
          </div>
        </div>
      </div>
    </div>

    <script>
      let selectedCloth = null;
      
      // Load cloth gallery
      fetch('/api/cloth-gallery')
        .then(r => r.json())
        .then(clothes => {
          const gallery = document.getElementById('cloth-gallery');
          clothes.forEach((cloth, idx) => {
            const item = document.createElement('div');
            item.className = 'cloth-item';
            item.innerHTML = `<img src="${cloth.url}" alt="Cloth ${idx+1}">`;
            item.onclick = () => {
              document.querySelectorAll('.cloth-item').forEach(i => i.classList.remove('selected'));
              item.classList.add('selected');
              selectedCloth = cloth.path;
              document.getElementById('cloth-preview').innerHTML = 
                `<img src="${cloth.url}" class="preview-img" alt="Selected cloth">`;
              checkReady();
            };
            gallery.appendChild(item);
          });
        });
      
      // File upload handlers
      ['person', 'cloth'].forEach(type => {
        const upload = document.getElementById(type + '-upload');
        const fileInput = document.getElementById(type + '-file');
        const preview = document.getElementById(type + '-preview');
        
        upload.onclick = () => fileInput.click();
        
        upload.ondragover = (e) => {
          e.preventDefault();
          upload.classList.add('dragover');
        };
        
        upload.ondragleave = () => upload.classList.remove('dragover');
        
        upload.ondrop = (e) => {
          e.preventDefault();
          upload.classList.remove('dragover');
          if (e.dataTransfer.files.length > 0) {
            fileInput.files = e.dataTransfer.files;
            handleFile(type, e.dataTransfer.files[0]);
          }
        };
        
        fileInput.onchange = (e) => {
          if (e.target.files.length > 0) {
            handleFile(type, e.target.files[0]);
          }
        };
      });
      
      function handleFile(type, file) {
        const preview = document.getElementById(type + '-preview');
        const reader = new FileReader();
        reader.onload = (e) => {
          preview.innerHTML = `<img src="${e.target.result}" class="preview-img" alt="${type} preview">`;
          if (type === 'cloth') selectedCloth = null;
          checkReady();
        };
        reader.readAsDataURL(file);
      }
      
      function checkReady() {
        const personFile = document.getElementById('person-file').files[0];
        const clothFile = document.getElementById('cloth-file').files[0];
        const btn = document.getElementById('tryon-btn');
        btn.disabled = !personFile && !selectedCloth || (!clothFile && !selectedCloth);
      }
      
      // Try-on button
      document.getElementById('tryon-btn').onclick = async () => {
        const personFile = document.getElementById('person-file').files[0];
        const clothFile = document.getElementById('cloth-file').files[0];
        
        if (!personFile) {
          alert('Please upload a person image');
          return;
        }
        
        if (!clothFile && !selectedCloth) {
          alert('Please upload or select a cloth image');
          return;
        }
        
        const formData = new FormData();
        formData.append('person', personFile);
        if (clothFile) {
          formData.append('cloth', clothFile);
        } else if (selectedCloth) {
          // Fetch cloth from server
          const clothBlob = await fetch('/api/cloth/' + encodeURIComponent(selectedCloth)).then(r => r.blob());
          formData.append('cloth', clothBlob, 'cloth.jpg');
        }
        
        const loading = document.getElementById('loading');
        const status = document.getElementById('status');
        const resultContainer = document.getElementById('result-container');
        const btn = document.getElementById('tryon-btn');
        
        loading.classList.add('active');
        status.className = 'status';
        btn.disabled = true;
        resultContainer.innerHTML = '';
        
        try {
          const response = await fetch('/tryon', {
            method: 'POST',
            body: formData
          });
          
          if (!response.ok) {
            throw new Error('Server error: ' + response.statusText);
          }
          
          const blob = await response.blob();
          const url = URL.createObjectURL(blob);
          resultContainer.innerHTML = `
            <h2 style="color: #667eea; margin-bottom: 20px;">✨ Try-On Result</h2>
            <img src="${url}" class="result-img" alt="Try-on result">
          `;
          status.className = 'status success';
          status.textContent = '✓ Try-on generated successfully!';
        } catch (err) {
          status.className = 'status error';
          status.textContent = '✗ Error: ' + err.message;
        } finally {
          loading.classList.remove('active');
          btn.disabled = false;
        }
      };
    </script>
  </body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    """Main dashboard UI."""
    return render_template_string(INDEX_HTML)

@app.route('/api/cloth-gallery', methods=['GET'])
def cloth_gallery():
    """Get list of available cloth images."""
    cloth_files = glob.glob(os.path.join(CLOTH_GALLERY_DIR, "*.jpg")) + \
                  glob.glob(os.path.join(CLOTH_GALLERY_DIR, "*.png"))
    
    clothes = []
    for cloth_path in cloth_files:
        rel_path = os.path.relpath(cloth_path, CLOTH_GALLERY_DIR)
        clothes.append({
            'path': rel_path,
            'url': f'/api/cloth/{rel_path}'
        })
    
    return jsonify(clothes)

@app.route('/api/cloth/<path:filename>', methods=['GET'])
def get_cloth(filename):
    """Serve cloth image."""
    cloth_path = os.path.join(CLOTH_GALLERY_DIR, filename)
    if os.path.exists(cloth_path):
        return send_file(cloth_path, mimetype='image/jpeg')
    return "Not found", 404

@app.route('/tryon', methods=['POST'])
def tryon():
    person = request.files.get('person')
    cloth = request.files.get('cloth')

    if person is None:
        return "Person image is required.", 400
    if cloth is None:
        return "Cloth image is required.", 400

    person_path = 'uploads/person.jpg'
    cloth_path = 'uploads/cloth.jpg'
    output_path = 'uploads/result.jpg'

    person.save(person_path)
    cloth.save(cloth_path)

    generate_tryon(person_path, cloth_path, output_path)

    return send_file(output_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
