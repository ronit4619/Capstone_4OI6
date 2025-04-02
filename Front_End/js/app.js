const API_URL = "http://localhost:5000"; // For login/register
const ANALYSIS_BACKEND_URL = "http://localhost:8001"; // Flask backend to trigger Python scripts

/****************************************
 *          Authentication Logic
 ****************************************/

function stopCamera() {
  let video = document.getElementById('videoFeed');
  if (video && video.srcObject) {
    let stream = video.srcObject;
    let tracks = stream.getTracks();
    tracks.forEach(track => track.stop());
    video.srcObject = null;
  }
}

function showRegister() {
  document.getElementById("loginBox").classList.add("hidden");
  document.getElementById("registerBox").classList.remove("hidden");
}

function showLogin() {
  document.getElementById("registerBox").classList.add("hidden");
  document.getElementById("loginBox").classList.remove("hidden");
}

async function login() {
  const username = document.getElementById("loginUsername").value;
  const password = document.getElementById("loginPassword").value;

  try {
    const response = await fetch(`${API_URL}/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });

    const result = await response.json();

    if (response.ok) {
      localStorage.setItem("jwtToken", result.token);
      localStorage.setItem("loggedInUser", username);

      alert("Login successful!");
      document.getElementById("authContainer").style.display = "none";
      document.getElementById("dashboard").style.display = "block";
      updateWelcomeMessage();
    } else {
      alert(result.message);
    }
  } catch (error) {
    console.error("Login error:", error);
  }
}

function updateWelcomeMessage() {
  const username = localStorage.getItem("loggedInUser");
  if (username) {
    document.getElementById("welcomeMessage").innerText = `Welcome, ${username}! 🏀`;
  } else {
    document.getElementById("welcomeMessage").innerText = "Welcome! 🏀";
  }
}

document.addEventListener("DOMContentLoaded", () => {
  if (localStorage.getItem("jwtToken")) {
    document.getElementById("authContainer").style.display = "none";
    document.getElementById("dashboard").style.display = "block";
    updateWelcomeMessage();
  } else {
    document.getElementById("authContainer").style.display = "flex";
    document.getElementById("dashboard").style.display = "none";
  }
});

function logout() {
  localStorage.removeItem("jwtToken");
  localStorage.removeItem("loggedInUser");

  stopCamera();
  clearUploadPreview();

  document.getElementById("authContainer").style.display = "flex";
  document.getElementById("dashboard").style.display = "none";
  document.getElementById("welcomeMessage").innerText = "Welcome! 🏀";

  if (document.getElementById("mainMenu")) {
    showMainMenu();
  }
}

/****************************************
 *          Registration Logic
 ****************************************/

async function register() {
  const username = document.getElementById("registerUsername").value;
  const email = document.getElementById("registerEmail").value;
  const password = document.getElementById("registerPassword").value;
  const confirmPass = document.getElementById("confirmPassword").value;
  const privacyPolicyChecked = document.getElementById("privacyPolicy").checked;
  const dob = document.getElementById("dob").value;

  if (!username || !password || !dob) {
    alert("Please fill in all required fields.");
    return;
  }

  if (password !== confirmPass) {
    alert("Passwords do not match!");
    return;
  }

  if (!privacyPolicyChecked) {
    alert("Please agree to the Privacy Policy to create an account.");
    return;
  }

  try {
    const response = await fetch(`${API_URL}/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, email, password, birthday: dob })
    });

    const result = await response.json();

    if (response.ok) {
      alert(result.message || "Registration successful! Please log in.");
      showLogin();
    } else {
      alert(result.message || "An error occurred.");
    }
  } catch (error) {
    console.error("Registration error:", error);
    alert("Error connecting to server.");
  }
}

/****************************************
 *          Navigation Logic
 ****************************************/

function showMainMenu() {
  document.getElementById("mainMenu").style.display = "flex";
  document.querySelectorAll(".page-content").forEach((p) =>
    p.classList.remove("active-page")
  );
  document.querySelector(".back-btn").classList.add("hidden");
}

function showPage(pageId) {
  document.getElementById("mainMenu").style.display = "none";
  document.getElementById(pageId).classList.add("active-page");
  document.querySelector(".back-btn").classList.remove("hidden");
}

/****************************************
 *          File Upload Logic
 ****************************************/

async function analyzeVideo() {
  const fileInput = document.getElementById("fileInput");
  const resultBox = document.getElementById("analysisResult");

  const file = fileInput.files[0];
  const formData = new FormData();
  formData.append("video", file);

  try {
    const res = await fetch(`${ANALYSIS_BACKEND_URL}/analyze`, {
      method: "POST",
      body: formData
    });

    const data = await res.json();
    resultBox.innerText = data.result;
  } catch (err) {
    resultBox.innerText = "Error analyzing video: " + err.message;
  }
}

document.getElementById("fileInput").addEventListener("change", function (e) {
  const file = e.target.files[0];
  const preview = document.getElementById("uploadPreview");
  const video = document.getElementById("uploadedVideo");
  const analyzeBtn = document.getElementById("analyzeButton");

  if (file && file.type.startsWith("video/")) {
    const url = URL.createObjectURL(file);
    video.src = url;
    preview.classList.remove("hidden");
    analyzeBtn.classList.remove("hidden");
  } else {
    alert("Please select a valid video file");
    preview.classList.add("hidden");
    analyzeBtn.classList.add("hidden");
  }
});

/****************************************
 *   Live Analysis Start/Stop Logic
 ****************************************/

function startAnalysis(type) {
  const loader = document.getElementById("loader");
  const videoImage = document.getElementById("liveStreamImage");

  loader.style.display = "block";
  videoImage.style.display = "none";

  fetch(`${ANALYSIS_BACKEND_URL}/start-analysis`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ type: type }),
  })
    .then((res) => res.json())
    .then((data) => {
      if (data.status === "ok") {
        setTimeout(() => {
          videoImage.src = "http://127.0.0.1:8002/video";
          videoImage.style.display = "block";
          loader.style.display = "none";
        }, 5000);
      } else {
        loader.style.display = "none";
        alert("Server error starting analysis.");
      }
    })
    .catch((err) => {
      loader.style.display = "none";
      console.error(err);
      alert("Server error starting analysis.");
    });
}


function stopAnalysis() {
  fetch(`${ANALYSIS_BACKEND_URL}/stop-analysis`, {
    method: "POST"
  })
  .then((res) => res.json())
  .then((data) => {
    if (data.status === "ok") {
      const frame = document.getElementById("liveStreamFrame");
      frame.style.display = "none";  // Hide the iframe
      frame.src = ""; // Clear the source to stop trying to reload
      alert("Analysis stopped.");
    } else {
      alert("No analysis was running.");
    }
  })
  .catch((err) => {
    console.error(err);
    alert("Failed to stop analysis.");
  });
}



/****************************************
 *  Other
 ****************************************/

function changePassword() {
  window.location.href = "change-password.html";
}

window.addEventListener("beforeunload", () => {
  stopCamera();
});

function navigateWithTransition(url) {
  const overlay = document.querySelector('.transition-overlay');
  overlay.style.transform = 'translateX(0)'; // Trigger the swipe effect

  // Wait for the transition to complete before navigating
  setTimeout(() => {
    window.location.href = url;
  }, 250); // Match the duration of the CSS transition
}
