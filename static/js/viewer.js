// Interactive MRI Slice Inspector with Grad-CAM Alpha Blending

class MRISliceViewer {
  constructor(canvasId, originalImgUrl, gradcamImgUrl) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext('2d');
    
    this.origImg = new Image();
    this.gradImg = new Image();
    
    this.origLoaded = false;
    this.gradLoaded = false;
    
    this.gradOpacity = 0.5;
    this.brightness = 100;
    this.contrast = 100;
    this.zoom = 1.0;
    this.showGradcam = true;
    
    this.loadImages(originalImgUrl, gradcamImgUrl);
    this.initControls();
  }
  
  loadImages(origUrl, gradUrl) {
    this.origLoaded = false;
    this.gradLoaded = false;
    
    this.origImg.onload = () => {
      this.origLoaded = true;
      this.render();
    };
    this.origImg.src = origUrl;
    
    if (gradUrl) {
      this.gradImg.onload = () => {
        this.gradLoaded = true;
        this.render();
      };
      this.gradImg.src = gradUrl;
    }
  }
  
  initControls() {
    const opacitySlider = document.getElementById('slider-gradcam-opacity');
    if (opacitySlider) {
      opacitySlider.addEventListener('input', (e) => {
        this.gradOpacity = parseFloat(e.target.value);
        this.render();
      });
    }
    
    const toggleGradBtn = document.getElementById('btn-toggle-gradcam');
    if (toggleGradBtn) {
      toggleGradBtn.addEventListener('click', () => {
        this.showGradcam = !this.showGradcam;
        toggleGradBtn.classList.toggle('active', this.showGradcam);
        this.render();
      });
    }
    
    const brightnessSlider = document.getElementById('slider-brightness');
    if (brightnessSlider) {
      brightnessSlider.addEventListener('input', (e) => {
        this.brightness = parseInt(e.target.value);
        this.render();
      });
    }
    
    const contrastSlider = document.getElementById('slider-contrast');
    if (contrastSlider) {
      contrastSlider.addEventListener('input', (e) => {
        this.contrast = parseInt(e.target.value);
        this.render();
      });
    }
  }
  
  render() {
    if (!this.origLoaded) return;
    
    const w = this.origImg.naturalWidth || 256;
    const h = this.origImg.naturalHeight || 256;
    
    this.canvas.width = w;
    this.canvas.height = h;
    
    this.ctx.clearRect(0, 0, w, h);
    
    // Apply CSS filters for brightness and contrast
    this.ctx.filter = `brightness(${this.brightness}%) contrast(${this.contrast}%)`;
    
    // Draw base MRI scan
    this.ctx.globalAlpha = 1.0;
    this.ctx.drawImage(this.origImg, 0, 0, w, h);
    
    // Draw Grad-CAM overlay if enabled
    if (this.showGradcam && this.gradLoaded && this.gradOpacity > 0) {
      this.ctx.filter = 'none';
      this.ctx.globalAlpha = this.gradOpacity;
      this.ctx.drawImage(this.gradImg, 0, 0, w, h);
    }
    
    this.ctx.globalAlpha = 1.0;
  }
}

window.MRISliceViewer = MRISliceViewer;
