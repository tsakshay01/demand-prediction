const sqlite3 = require('sqlite3').verbose();
const path = require('path');
const bcrypt = require('bcrypt');

const dbPath = path.resolve(__dirname, 'demand_ai.db');
const db = new sqlite3.Database(dbPath);

// Initialize DB Tables
db.serialize(() => {
    // Users Table
    db.run(`CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE,
        password_hash TEXT,
        name TEXT,
        role TEXT DEFAULT 'analyst',
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )`);

    // Files Table (Metadata for uploads)
    db.run(`CREATE TABLE IF NOT EXISTS uploaded_files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        original_name TEXT,
        filename TEXT,
        size INTEGER,
        uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )`);

    // Ensure Admin Exists (or update existing user to admin)
    const adminEmail = 'admin@demand.ai';
    db.get("SELECT id, role FROM users WHERE email = ?", [adminEmail], async (err, row) => {
        if (!row) {
            // Create new admin
            const hash = await bcrypt.hash('admin123', 10);
            const stmt = db.prepare("INSERT INTO users (email, password_hash, name, role) VALUES (?, ?, ?, ?)");
            stmt.run(adminEmail, hash, 'Admin', 'admin');
            stmt.finalize();
            console.log("✅ Default Admin Created: admin@demand.ai / admin123");
        } else if (row.role !== 'admin') {
            // Update existing user to admin role
            db.run("UPDATE users SET role = 'admin', name = 'Admin' WHERE email = ?", [adminEmail]);
            console.log("✅ Updated admin@demand.ai to admin role");
        }
    });

    console.log("✅ SQLite Database Connected & Initialized");
});

module.exports = db;
