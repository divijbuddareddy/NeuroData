// Upload and Batch Processing Handler

document.addEventListener('DOMContentLoaded', () => {
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');
  const fileListContainer = document.getElementById('selected-files-list');
  const btnStartAnalysis = document.getElementById('btn-start-upload-analysis');
  const datasetNameInput = document.getElementById('dataset-name-input');
  
  let selectedFiles = [];
  
  if (!dropZone || !fileInput) return;
  
  ['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropZone.classList.add('drag-over');
    });
  });
  
  ['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropZone.classList.remove('drag-over');
    });
  });
  
  dropZone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = Array.from(dt.files);
    handleFilesAdded(files);
  });
  
  fileInput.addEventListener('change', (e) => {
    const files = Array.from(e.target.files);
    handleFilesAdded(files);
  });
  
  function handleFilesAdded(files) {
    selectedFiles = [...selectedFiles, ...files];
    renderFileList();
  }
  
  function renderFileList() {
    if (!fileListContainer) return;
    if (selectedFiles.length === 0) {
      fileListContainer.innerHTML = `<div style="color:var(--text-muted);font-size:12.5px;text-align:center;padding:16px;">No files selected yet.</div>`;
      if (btnStartAnalysis) btnStartAnalysis.disabled = true;
      return;
    }
    
    if (btnStartAnalysis) btnStartAnalysis.disabled = false;
    
    fileListContainer.innerHTML = selectedFiles.map((f, i) => `
      <div style="display:flex;align-items:center;justify-content:space-between;padding:8px 12px;background:var(--bg-dark);border-radius:6px;border:1px solid var(--border-subtle);font-size:12.5px;">
        <span style="font-family:var(--font-mono);">${f.name}</span>
        <span style="color:var(--text-muted);font-size:11px;">${(f.size/1024).toFixed(1)} KB</span>
      </div>
    `).join('');
  }
  
  if (btnStartAnalysis) {
    btnStartAnalysis.addEventListener('click', async () => {
      if (selectedFiles.length === 0) return;
      
      const formData = new FormData();
      selectedFiles.forEach(f => formData.append('files[]', f));
      formData.append('dataset_name', datasetNameInput ? datasetNameInput.value.trim() : 'Custom_Cohort');
      
      btnStartAnalysis.disabled = true;
      btnStartAnalysis.innerHTML = `<span class="animate-spin">⏳</span> Uploading & Analyzing...`;
      
      try {
        const uploadRes = await fetch('/api/upload', {
          method: 'POST',
          body: formData
        });
        const uploadData = await uploadRes.json();
        
        if (!uploadRes.ok) {
          showToast(uploadData.error || "Upload failed", "error");
          btnStartAnalysis.disabled = false;
          btnStartAnalysis.innerHTML = "Start AI/ML Quality Screening";
          return;
        }
        
        showToast("Files uploaded. Running PyTorch & Scikit-learn quality pipeline...", "info");
        
        const analyzeRes = await fetch('/api/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            dataset_dir: uploadData.dataset_dir,
            dataset_name: uploadData.dataset_name
          })
        });
        
        const analyzeData = await analyzeRes.json();
        if (analyzeRes.ok && analyzeData.dataset_id) {
          showToast("Dataset successfully analyzed!", "success");
          setTimeout(() => {
            window.location.href = `/scan-quality?dataset_id=${analyzeData.dataset_id}`;
          }, 600);
        } else {
          showToast(analyzeData.error || "Analysis pipeline failed", "error");
          btnStartAnalysis.disabled = false;
          btnStartAnalysis.innerHTML = "Start AI/ML Quality Screening";
        }
      } catch (err) {
        showToast("Error: " + err.message, "error");
        btnStartAnalysis.disabled = false;
        btnStartAnalysis.innerHTML = "Start AI/ML Quality Screening";
      }
    });
  }
});
