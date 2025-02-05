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
 *          Helper Function
 ****************************************/
function calculateAge(dobString) {
    const today = new Date();
    const birthDate = new Date(dobString);
  
    let age = today.getFullYear() - birthDate.getFullYear();
    const monthDiff = today.getMonth() - birthDate.getMonth();
  
    // Check if the birthday has not happened yet this year
    if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < birthDate.getDate())) {
      age--;
    }
    return age;
  }
  
  /****************************************
   *          Registration Logic
   ****************************************/
  function register() {
    const username = document.getElementById('registerUsername').value;
    const password = document.getElementById('registerPassword').value;
    const confirmPass = document.getElementById('confirmPassword').value;
    const privacyPolicyChecked = document.getElementById('privacyPolicy').checked;
    const dob = document.getElementById('dob').value;
  
    // Calculate user age
    const userAge = calculateAge(dob);
  
    // Check if the user is under 13
    if (userAge < 13) {
      // Reveal the parental consent section
      document.getElementById('parentConsentSection').classList.remove('hidden');
  
      // Check if the parent consent was provided
      const parentConsentChecked = document.getElementById('parentConsent').checked;
      if (!parentConsentChecked) {
        alert('Users under 13 need parental consent.');
        return;
      }
    }
  
    // Check if privacy policy is agreed
    if (!privacyPolicyChecked) {
      alert('Please agree to the Privacy Policy to create an account.');
      return;
    }
  
    // Verify password match
    if (password !== confirmPass) {
      alert('Passwords do not match!');
      return;
    }
  
    // Check if username already exists
    if (users.some(user => user.username === username)) {
      alert('Username already exists!');
      return;
    }
  
    // Create the new user object
    users.push({ 
      username, 
      password, 
      dob, 
      age: userAge,
      // For demonstration, store parent's consent status if needed
    });
    
    localStorage.setItem('basketballUsers', JSON.stringify(users));
    alert('Registration successful!');
    showLogin();
  }
  
  /****************************************
   *          UI Events
   ****************************************/
  document.getElementById('dob').addEventListener('change', function() {
    const dobValue = this.value;
    const userAge = calculateAge(dobValue);
  
    // If user is under 13, show parental consent
    if (userAge < 13) {
      document.getElementById('parentConsentSection').classList.remove('hidden');
    } else {
      // Hide the parental consent section if they are >= 13
      document.getElementById('parentConsentSection').classList.add('hidden');
      // Optionally uncheck or clear parent's email
      document.getElementById('parentConsent').checked = false;
      // document.getElementById('parentEmail').value = '';
    }
  });
  

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
