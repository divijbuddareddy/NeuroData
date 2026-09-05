// Gemini AI Interactive Real-Time Research Assistant Logic

document.addEventListener('DOMContentLoaded', () => {
  const chatMessages = document.getElementById('chat-messages');
  const chatInput = document.getElementById('chat-input');
  const chatSendBtn = document.getElementById('chat-send-btn');
  const datasetSelect = document.getElementById('assistant-dataset-select');
  
  if (!chatMessages || !chatInput || !chatSendBtn) return;
  
  function appendMessage(role, text) {
    const bubble = document.createElement('div');
    bubble.className = `chat-bubble ${role}`;
    
    // Markdown-like paragraph and bold formatting
    let formatted = text
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/`([^`]+)`/g, '<code style="background:rgba(0,0,0,0.08);padding:2px 4px;border-radius:4px;font-family:var(--font-mono);font-size:12px;">$1</code>')
      .replace(/\n\n/g, '<br><br>')
      .replace(/\n/g, '<br>');
      
    bubble.innerHTML = `<strong>${role === 'user' ? 'You' : 'Gemini 2.5 Flash'}:</strong><br>${formatted}`;
    
    chatMessages.appendChild(bubble);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }
  
  async function sendMessage() {
    const text = chatInput.value.trim();
    if (!text) return;
    
    appendMessage('user', text);
    chatInput.value = '';
    
    const datasetId = datasetSelect ? datasetSelect.value : null;
    const activeApiKey = localStorage.getItem('google_studio_api_key') || '';
    
    // Typing indicator
    const typingIndicator = document.createElement('div');
    typingIndicator.className = 'chat-bubble assistant';
    typingIndicator.id = 'typing-indicator';
    typingIndicator.innerHTML = `<em>Gemini 2.5 Flash is thinking & synthesizing answer in real-time...</em>`;
    chatMessages.appendChild(typingIndicator);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'X-Gemini-Api-Key': activeApiKey
        },
        body: JSON.stringify({
          message: text,
          dataset_id: datasetId ? parseInt(datasetId) : null,
          api_key: activeApiKey
        })
      });
      
      const data = await res.json();
      typingIndicator.remove();
      
      if (res.ok && data.reply) {
        appendMessage('assistant', data.reply);
      } else {
        appendMessage('assistant', "Encountered an issue: " + (data.error || "Unknown error"));
      }
    } catch (err) {
      typingIndicator.remove();
      appendMessage('assistant', "Network connection error: " + err.message);
    }
  }
  
  chatSendBtn.addEventListener('click', sendMessage);
  chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
  
  // Suggested query chips
  document.querySelectorAll('.suggested-prompt-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      chatInput.value = chip.getAttribute('data-prompt');
      sendMessage();
    });
  });
});
