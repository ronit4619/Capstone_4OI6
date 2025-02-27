const API_URL = "http://localhost:5000"; // Adjust if needed

document.getElementById("deleteAccountForm").addEventListener("submit", async function (e) {
    e.preventDefault();

    const password = document.getElementById("password").value;
    const confirmPassword = document.getElementById("confirmPassword").value;
    const confirmCheckbox = document.getElementById("confirmCheckbox").checked;
    const token = localStorage.getItem("jwtToken");

    if (password !== confirmPassword) {
        alert("Passwords do not match!");
        return;
    }

    if (!confirmCheckbox) {
        alert("You must confirm that you understand account deletion is permanent.");
        return;
    }

    const finalConfirmation = confirm("Are you absolutely sure you want to delete your account? This cannot be undone.");
    if (!finalConfirmation) {
        return;
    }

    try {
        console.log("Sending Delete Request to:", `${API_URL}/delete-account`);

        const response = await fetch(`${API_URL}/delete-account`, {
            method: "DELETE",
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${token}`
            },
            body: JSON.stringify({ password }),
        });

        const result = await response.json();
        alert(result.message);

        if (response.ok) {
            localStorage.removeItem("jwtToken"); // Clear user session
            window.location.href = result.redirect || "index.html"; // Redirect to sign-in page
        }
    } catch (error) {
        console.error("Error deleting account:", error);
        alert("Something went wrong. Try again.");
    }
});
