const API_URL = "http://localhost:5000"; // Adjust based on your backend
document.getElementById("changePasswordForm").addEventListener("submit", async function (e) {
    e.preventDefault();

    const oldPassword = document.getElementById("oldPassword").value;
    const newPassword = document.getElementById("newPassword").value;
    const confirmNewPassword = document.getElementById("confirmNewPassword").value;
    
    if (newPassword !== confirmNewPassword) {
        alert("New passwords do not match!");
        return;
    }

    const token = localStorage.getItem("jwtToken");

    console.log("Token being sent:", token); // ✅ Debugging

    if (!token) {
        alert("You must be logged in to change your password.");
        return;
    }

    try {
        const response = await fetch(`${API_URL}/change-password`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${token}`
            },
            body: JSON.stringify({ oldPassword, newPassword }),
        });

        const result = await response.json();
        console.log("Response:", result); // ✅ Debugging

        if (response.ok) {
            alert(result.message);
            window.location.href = result.redirect || "index.html"; // ✅ Redirect after success
        } else {
            alert(result.message);
        }
    } catch (error) {
        console.error("Error changing password:", error);
        alert("Something went wrong. Try again.");
    }
});
