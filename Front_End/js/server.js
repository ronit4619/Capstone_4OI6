require("dotenv").config();
const express = require("express");
const mysql = require("mysql2");
const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");
const cors = require("cors");
const bodyParser = require("body-parser");

const app = express();
app.use(cors());
app.use(bodyParser.json());

// MySQL Database Connection
const db = mysql.createConnection({
  host: process.env.DB_HOST,
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  database: process.env.DB_NAME,
});

db.connect((err) => {
  if (err) {
    console.error("❌ Database connection failed:", err.message);
  } else {
    console.log("✅ Connected to MySQL database.");
  }
});

// JWT Secret Key
const JWT_SECRET = process.env.JWT_SECRET || "your_jwt_secret"; 
console.log("Loaded JWT_SECRET:", JWT_SECRET);


// Hash password function
const hashPassword = async (password) => {
  const salt = await bcrypt.genSalt(10);
  return await bcrypt.hash(password, salt);
};

// Middleware to verify JWT Token
const authenticateToken = (req, res, next) => {
  const authHeader = req.header("Authorization");
  if (!authHeader || !authHeader.startsWith("Bearer ")) {
    return res.status(401).json({ message: "Access Denied: No token provided" });
  }

  const token = authHeader.split(" ")[1];

  console.log("Extracted Token:", token); // Debugging

  jwt.verify(token, JWT_SECRET, (err, user) => {
    if (err) {
      console.error("JWT Verification Error:", err); // Debugging
      return res.status(403).json({ message: "Invalid Token" });
    }

    console.log("Decoded User:", user); // Debugging
    req.user = user;
    next();
  });
};


// 📝 **Register a User (Using `storeUser` Procedure)**
app.post("/register", async (req, res) => {
    const { username, email = "", password, birthday } = req.body; // Default email to empty string if missing

    if (!username || !password || !birthday) {
        return res.status(400).json({ message: "Username, password, and birthday are required." });
    }

    try {
        const hashedPassword = await hashPassword(password);

        db.query("CALL storeUser(?, ?, ?, ?)", [username, hashedPassword, email, birthday], (err, result) => {
            if (err) {
                console.error("Database Error:", err);
                return res.status(500).json({ message: "Database error: " + err.message });
            }
            res.status(201).json({ message: "User registered successfully!" });
        });

    } catch (error) {
        console.error("Server error:", error);
        res.status(500).json({ message: "Internal server error" });
    }
});

  

app.post("/login", (req, res) => {
  const { username, password } = req.body;

  db.query("CALL getUserByUsername(?)", [username], async (err, results) => {
    if (err) return res.status(500).json({ error: err.message });

    if (results[0].length === 0) {
      return res.status(400).json({ message: "Invalid credentials" });
    }

    const user = results[0][0];
    const isMatch = await bcrypt.compare(password, user.password);

    if (!isMatch) return res.status(400).json({ message: "Invalid credentials" });

    // Generate JWT Token
    const token = jwt.sign({ userID: user.userID, username: user.username }, JWT_SECRET, {
      expiresIn: "1h",
    });

    console.log("Generated Token:", token); // ✅ Debugging

    res.json({ message: "Login successful", token });
  });
});


// 🔒 **Get User Profile (Protected Route)**
app.get("/profile", authenticateToken, (req, res) => {
  db.query("SELECT username, birthday FROM users WHERE userID = ?", [req.user.userID], (err, results) => {
    if (err) return res.status(500).json({ error: err.message });

    res.json(results[0]);
  });
});

// ✅ API Route for Changing Password
app.post("/change-password", authenticateToken, (req, res) => {
    const { oldPassword, newPassword } = req.body;
    const userId = req.user.userID; // Ensure this matches your stored procedure

    console.log("User ID Received:", userId);

    if (!oldPassword || !newPassword) {
        return res.status(400).json({ message: "Both old and new passwords are required." });
    }

    // Fetch stored hashed password
    db.query("SELECT password FROM users WHERE userID = ?", [userId], async (err, results) => {
        if (err) {
            console.error("❌ Database Error:", err);
            return res.status(500).json({ message: "Database error." });
        }

        if (results.length === 0) {
            return res.status(404).json({ message: "User not found." });
        }

        const storedPassword = results[0].password;

        try {
            // Compare old password with stored bcrypt hash
            const isMatch = await bcrypt.compare(oldPassword, storedPassword);
            if (!isMatch) {
                return res.status(400).json({ message: "Old password is incorrect." });
            }

            // Hash the new password
            const hashedNewPassword = await bcrypt.hash(newPassword, 10);

            // Update password using the stored procedure
            db.query("CALL changePassword(?, ?)", [userId, hashedNewPassword], (err, result) => {
                if (err) {
                    console.error("❌ Error updating password:", err);
                    return res.status(500).json({ message: "Error updating password." });
                }

                console.log("✅ Password changed successfully!");
                return res.json({ message: "Password changed successfully!", redirect: "file:///C:/Users/antho/OneDrive/Desktop/Year%205/Capstone/Capstone_4OI6/Front_End/index.html" }); /// have to be changed in the futue
            });

        } catch (error) {
            console.error("❌ Server Error:", error);
            return res.status(500).json({ message: "Internal server error." });
        }
    });
});


//🔒 **Delete Account (Protected Route)** 

app.delete("/delete-account", authenticateToken, async (req, res) => {
    const { password } = req.body;
    const userId = req.user.userID; // Ensure correct user is deleting their account

    console.log("User ID Requesting Deletion:", userId);

    if (!password) {
        return res.status(400).json({ message: "Password is required for account deletion." });
    }

    try {
        // Retrieve stored hashed password
        db.query("SELECT password FROM users WHERE userID = ?", [userId], async (err, results) => {
            if (err) {
                console.error("❌ Database Error:", err);
                return res.status(500).json({ message: "Database error." });
            }

            if (results.length === 0) {
                return res.status(404).json({ message: "User not found." });
            }

            const storedPassword = results[0].password;

            // Compare entered password with stored bcrypt hash
            const isMatch = await bcrypt.compare(password, storedPassword);
            if (!isMatch) {
                return res.status(400).json({ message: "Incorrect password. Account not deleted." });
            }

            // Call the `deleteAccount` stored procedure
            db.query("CALL deleteAccount(?)", [userId], (err, result) => {
                if (err) {
                    console.error("❌ Error deleting account:", err);
                    return res.status(500).json({ message: "Error deleting account." });
                }

                console.log("✅ Account deleted successfully!");
                res.json({ message: "Your account has been deleted successfully.", redirect: "index.html" });
            });
        });

    } catch (error) {
        console.error("❌ Server Error:", error);
        res.status(500).json({ message: "Internal server error." });
    }
});


//🚀 Start Server
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`✅ Server running on http://localhost:${PORT}`);
});
