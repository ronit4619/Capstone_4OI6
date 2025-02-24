const API_URL = "http://localhost:5000"; // Change if running on a different host

/****************************************
 *          Authentication Logic
 ****************************************/
function showRegister() {
  document.getElementById("loginBox").classList.add("hidden");
  document.getElementById("registerBox").classList.remove("hidden");
}

function showLogin() {
  document.getElementById("registerBox").classList.add("hidden");
  document.getElementById("loginBox").classList.remove("hidden");
}

// ✅ Login Function
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
      alert("Login successful!");
      document.getElementById("authContainer").style.display = "none";
      document.getElementById("dashboard").style.display = "block";
      showMainMenu();
    } else {
      alert(result.message);
    }
  } catch (error) {
    console.error("Login error:", error);
  }
}

// ✅ Logout Function
function logout() {
  localStorage.removeItem("jwtToken");
  stopCamera();
  clearUploadPreview();
  document.getElementById("authContainer").style.display = "flex";
  document.getElementById("dashboard").style.display = "none";
  showMainMenu();
}

/****************************************
 *          Registration Logic
 ****************************************/
async function register() {
  const username = document.getElementById("registerUsername").value;
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
    const response = await fetch("http://localhost:5000/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password, birthday: dob }),
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
document.getElementById("fileInput").addEventListener("change", function (e) {
  const file = e.target.files[0];
  const preview = document.getElementById("uploadPreview");
  const video = document.getElementById("uploadedVideo");

  if (file && file.type.startsWith("video/")) {
    const url = URL.createObjectURL(file);
    video.src = url;
    preview.classList.remove("hidden");
  } else {
    alert("Please select a valid video file");
  }
});

function clearUploadPreview() {
  const preview = document.getElementById("uploadPreview");
  const video = document.getElementById("uploadedVideo");
  preview.classList.add("hidden");
  video.src = "";
  document.getElementById("fileInput").value = "";
}

/****************************************
 *  Stop camera on page unload (optional)
 ****************************************/
window.addEventListener("beforeunload", () => {
  stopCamera();
});
