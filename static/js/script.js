// script.js
const chatBox = document.getElementById("chatBox");
const messageInput = document.getElementById("messageInput");
const fileUpload = document.getElementById("fileUpload");

// Display system message at start
function addSystemMessage(message) {
  const msgDiv = document.createElement("div");
  msgDiv.className = "message bot";
  msgDiv.textContent = message;

  const timeStamp = document.createElement("div");
  timeStamp.className = "time-stamp";
  timeStamp.textContent = getCurrentTime();
  msgDiv.appendChild(timeStamp);

  chatBox.appendChild(msgDiv);
  chatBox.scrollTop = chatBox.scrollHeight;
}

// Initial system message
addSystemMessage("Hello! I can help you with general questions or analyze PDF documents. You can upload a PDF or just ask me anything!");

// Allow sending message with Enter key
messageInput.addEventListener("keypress", function(event) {
  if (event.key === "Enter") {
    event.preventDefault();
    sendMessage();
  }
});

// Handle file uploads separately
fileUpload.addEventListener("change", async function() {
  const file = fileUpload.files[0];
  if (!file) return;
  
  if (!file.name.endsWith('.pdf')) {
    addSystemMessage("Only PDF files are supported");
    fileUpload.value = "";
    return;
  }
  
  // Show file upload message
  const msgDiv = document.createElement("div");
  msgDiv.className = "message user";
  msgDiv.textContent = `Uploading document: ${file.name}`;
  
  const timeStamp = document.createElement("div");
  timeStamp.className = "time-stamp";
  timeStamp.textContent = getCurrentTime();
  msgDiv.appendChild(timeStamp);
  
  chatBox.appendChild(msgDiv);
  chatBox.scrollTop = chatBox.scrollHeight;
  
  // Show loading indicator
  const loadingIndicator = document.createElement("div");
  loadingIndicator.className = "message bot";
  loadingIndicator.textContent = "Processing document...";
  chatBox.appendChild(loadingIndicator);
  chatBox.scrollTop = chatBox.scrollHeight;
  
  try {
    const formData = new FormData();
    formData.append("file", file);
    
    const response = await fetch("/api/upload", {
      method: "POST",
      body: formData
    });
    
    const data = await response.json();
    
    // Remove loading indicator
    chatBox.removeChild(loadingIndicator);
    
    // Show success message
    addSystemMessage(`Document loaded successfully! You can now ask questions about ${file.name}`);
  } catch (error) {
    // Remove loading indicator
    chatBox.removeChild(loadingIndicator);
    
    // Show error message
    addSystemMessage("Error processing the document. Please try another file.");
  }
  
  // Reset file input
  fileUpload.value = "";
});

function getCurrentTime() {
  const now = new Date();
  return now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function getSourceIndicator(source) {
  switch(source) {
    case 'pdf':
      return '📄 PDF Answer';
    case 'general':
      return '🤖 General Answer';
    case 'summary':
      return '📝 Summary';
    default:
      return '';
  }
}

async function sendMessage() {
  const messageText = messageInput.value.trim();
  const file = fileUpload.files[0];

  if (!messageText && !file) return;

  // Show user message
  const msgDiv = document.createElement("div");
  msgDiv.className = "message user";
  msgDiv.textContent = messageText;

  const timeStamp = document.createElement("div");
  timeStamp.className = "time-stamp";
  timeStamp.textContent = getCurrentTime();
  msgDiv.appendChild(timeStamp);

  chatBox.appendChild(msgDiv);
  chatBox.scrollTop = chatBox.scrollHeight;

  // Reset text input
  messageInput.value = "";

  // Show typing indicator
  const typingIndicator = document.createElement("div");
  typingIndicator.className = "message bot";
  typingIndicator.textContent = "Thinking...";
  chatBox.appendChild(typingIndicator);
  chatBox.scrollTop = chatBox.scrollHeight;

  try {
    // Send to backend
    const formData = new FormData();
    formData.append("message", messageText);
    if (file) formData.append("file", file);

    const response = await fetch("/api/chat", {
      method: "POST",
      body: formData
    });

    const data = await response.json();

    // Remove typing indicator
    chatBox.removeChild(typingIndicator);

    // Show bot reply
    const botMsg = document.createElement("div");
    botMsg.className = "message bot";
    
    // Add source indicator if available
    if (data.source) {
      const sourceIndicator = document.createElement("div");
      sourceIndicator.className = "source-indicator";
      sourceIndicator.textContent = getSourceIndicator(data.source);
      botMsg.appendChild(sourceIndicator);
    }
    
    // Add the message content
    const messageContent = document.createElement("div");
    messageContent.className = "message-content";
    messageContent.textContent = data.reply;
    botMsg.appendChild(messageContent);
    
    const botTimeStamp = document.createElement("div");
    botTimeStamp.className = "time-stamp";
    botTimeStamp.textContent = getCurrentTime();
    botMsg.appendChild(botTimeStamp);
    
    chatBox.appendChild(botMsg);
  } catch (error) {
    // Remove typing indicator
    chatBox.removeChild(typingIndicator);
    
    // Show error message
    const errorMsg = document.createElement("div");
    errorMsg.className = "message bot";
    errorMsg.textContent = "Sorry, there was an error processing your request. Please try again.";
    
    const errorTimeStamp = document.createElement("div");
    errorTimeStamp.className = "time-stamp";
    errorTimeStamp.textContent = getCurrentTime();
    errorMsg.appendChild(errorTimeStamp);
    
    chatBox.appendChild(errorMsg);
  }
  
  // Reset file input if any
  fileUpload.value = "";
  
  chatBox.scrollTop = chatBox.scrollHeight;
}