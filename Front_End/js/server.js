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

// Hash password function
const hashPassword = async (password) => {
  const salt = await bcrypt.genSalt(10);
  return await bcrypt.hash(password, salt);
};

// Middleware to verify JWT Token
const authenticateToken = (req, res, next) => {
  const token = req.header("Authorization");
  if (!token) return res.status(401).json({ message: "Access Denied" });

  jwt.verify(token, JWT_SECRET, (err, user) => {
    if (err) return res.status(403).json({ message: "Invalid Token" });
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

  

// 🔐 **Login a User (Using `getUserByUsername` Procedure)**
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

// 🚀 Start Server
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`✅ Server running on http://localhost:${PORT}`);
});
