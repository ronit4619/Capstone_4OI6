/****************************************
 *          Global Variables
 ****************************************/
let users = JSON.parse(localStorage.getItem('basketballUsers')) || [];
let cameraStream = null;

/****************************************
 *          Authentication Logic
 ****************************************/
function showRegister() {
  document.getElementById('loginBox').classList.add('hidden');
  document.getElementById('registerBox').classList.remove('hidden');
}

function showLogin() {
    document.getElementById('registerBox').classList.add('hidden');
    document.getElementById('loginBox').classList.remove('hidden');
}

function register() {
    const username = document.getElementById('registerUsername').value;
    const password = document.getElementById('registerPassword').value;
    const confirmPass = document.getElementById('confirmPassword').value;
    const privacyPolicyChecked = document.getElementById('privacyPolicy').checked;
  
    // Double-check the policy
    if (!privacyPolicyChecked) {
      alert('Please agree to the Privacy Policy to create an account.');
      return;
    }
  
    if (password !== confirmPass) {
      alert('Passwords do not match!');
      return;
    }
  
    if (users.some(user => user.username === username)) {
      alert('Username already exists!');
      return;
    }
  
    users.push({ username, password });
    localStorage.setItem('basketballUsers', JSON.stringify(users));
    alert('Registration successful!');
    showLogin();
  }  
  

function login() {
  const username = document.getElementById('loginUsername').value;
  const password = document.getElementById('loginPassword').value;

  const user = users.find(u => u.username === username && u.password === password);

  if (user) {
    document.getElementById('authContainer').style.display = 'none';
    document.getElementById('dashboard').style.display = 'block';
    showMainMenu();
  } else {
    alert('Invalid credentials!');
  }
}

function logout() {
  stopCamera();
  clearUploadPreview();
  document.getElementById('authContainer').style.display = 'flex';
  document.getElementById('dashboard').style.display = 'none';
  showMainMenu();
}

/****************************************
 *          Navigation Logic
 ****************************************/
function showMainMenu() {
  document.getElementById('mainMenu').style.display = 'flex';
  document.querySelectorAll('.page-content').forEach(p => p.classList.remove('active-page'));
  document.querySelector('.back-btn').classList.add('hidden');
}

function showPage(pageId) {
  document.getElementById('mainMenu').style.display = 'none';
  document.getElementById(pageId).classList.add('active-page');
  document.querySelector('.back-btn').classList.remove('hidden');
}

/****************************************
 *          Camera Logic
 ****************************************/
async function toggleCamera() {
  const video = document.getElementById('videoFeed');
  const toggleBtn = document.getElementById('cameraToggle');

  if (!cameraStream) {
    try {
      cameraStream = await navigator.mediaDevices.getUserMedia({ video: true });
      video.srcObject = cameraStream;
      toggleBtn.textContent = 'Stop Camera';
    } catch (err) {
      alert('Error accessing camera: ' + err.message);
    }
  } else {
    stopCamera();
    toggleBtn.textContent = 'Start Camera';
  }
}

function stopCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach(track => track.stop());
    cameraStream = null;
    document.getElementById('videoFeed').srcObject = null;
  }
}

/****************************************
 *          File Upload Logic
 ****************************************/
document.getElementById('fileInput').addEventListener('change', function(e) {
  const file = e.target.files[0];
  const preview = document.getElementById('uploadPreview');
  const video = document.getElementById('uploadedVideo');

  if (file && file.type.startsWith('video/')) {
    const url = URL.createObjectURL(file);
    video.src = url;
    preview.classList.remove('hidden');
  } else {
    alert('Please select a valid video file');
  }
});

function clearUploadPreview() {
  const preview = document.getElementById('uploadPreview');
  const video = document.getElementById('uploadedVideo');
  preview.classList.add('hidden');
  video.src = '';
  document.getElementById('fileInput').value = '';
}

/****************************************
 *  Stop camera on page unload (optional)
 ****************************************/
window.addEventListener('beforeunload', () => {
  stopCamera();
});
