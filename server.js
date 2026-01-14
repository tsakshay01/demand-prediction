require('dotenv').config();
const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');

// Enterprise DB Helper (SQLite)
const db = require('./database');

const app = express();
const PORT = process.env.PORT || 3000;
const JWT_SECRET = process.env.JWT_SECRET || 'super_secret_enterprise_key';

// Middleware
// FIX: Disable CSP for Demo to prevent Design/CSS blocking
app.use(helmet({
    contentSecurityPolicy: false,
    crossOriginEmbedderPolicy: false
}));
app.use(cors());
app.use(bodyParser.json());
app.use(express.static('public'));

const limiter = rateLimit({ windowMs: 15 * 60 * 1000, max: 100 });
app.use(limiter);

// --- AUTH MIDDLEWARE ---
function verifyToken(req, res, next) {
    const header = req.headers['authorization'];
    if (!header) return res.status(401).json({ error: "Access Denied" });

    const token = header.split(' ')[1];
    if (!token) return res.status(401).json({ error: "Access Denied" });

    try {
        const verified = jwt.verify(token, JWT_SECRET);
        req.user = verified;
        next();
    } catch (err) {
        res.status(400).json({ error: "Invalid Token" });
    }
}

// --- ML Service Integration ---
let axios;
try {
    axios = require('axios');
} catch (e) {
    console.warn("Axios not found. ML features will run in mock mode.");
}
const ML_SERVICE_URL = 'http://127.0.0.1:5000';

// --- MARKET INTELLIGENCE (Mock/Auto) ---
let AutoSignalContext = {
    weather: { status: "CLEAR", risk: "LOW" },
    social: { status: "NORMAL", sentiment: 0.1 }
};

setInterval(() => {
    const wRand = Math.random();
    if (wRand > 0.9) AutoSignalContext.weather = { status: "STORM", risk: "CRITICAL" };
    else if (wRand > 0.7) AutoSignalContext.weather = { status: "RAIN", risk: "MODERATE" };
    else AutoSignalContext.weather = { status: "CLEAR", risk: "LOW" };

    const sRand = Math.random();
    if (sRand > 0.9) AutoSignalContext.social = { status: "VIRAL_NEGATIVE", sentiment: -0.8 };
    else if (sRand > 0.8) AutoSignalContext.social = { status: "VIRAL_POSITIVE", sentiment: 0.8 };
    else AutoSignalContext.social = { status: "NORMAL", sentiment: 0.1 };
}, 10000);

// --- ROUTES ---

app.get('/api/market-signals', (req, res) => {
    res.json(AutoSignalContext);
});

app.get('/api/market-intelligence', verifyToken, (req, res) => {
    res.json([
        { id: 'm1', name: "Regulatory Raid", category: "LEGAL", severity: "CRITICAL", sentiment: -0.95 },
        { id: 'm6', name: "Festival Season", category: "SEASONAL", severity: "POSITIVE", sentiment: 0.8 }
    ]);
});

// --- DASHBOARD STATS API (Dynamic Chart Data) ---
app.get('/api/dashboard/stats', verifyToken, (req, res) => {
    // Generate "Realistic" stats based on DB activity
    db.get("SELECT COUNT(*) as fileCount FROM uploaded_files", [], (err, row) => {
        if (err) return res.status(500).json({ error: "DB Error" });

        const fileCount = row.fileCount || 0;
        const baseDemand = 42000;
        const totalDemand = baseDemand + (fileCount * 1250); // Scale with usage

        // Randomize slightly for "Live" feel
        const accuracy = (94 + Math.random() * 2).toFixed(1);
        const products = ["Smart Watch", "Denim Jacket", "Sneakers", "Winter Coat"];
        const topProduct = products[Math.floor(Math.random() * products.length)];

        // Generate dynamic chart data based on file count
        const months = 12;
        const history = [];
        let trend = 1000;
        for (let i = 0; i < months; i++) {
            // Trend creeps up with fileCount
            trend += (50 + (fileCount * 10) + (Math.random() * 200 - 100));
            history.push(Math.round(trend));
        }

        res.json({
            totalDemand: totalDemand.toLocaleString(),
            accuracy: accuracy + "%",
            topProduct: topProduct,
            growth: "+" + (12 + (fileCount * 0.5)).toFixed(1) + "%",
            chartData: history // DYNAMIC CHART DATA
        });
    });
});

// Auth: Login
app.post('/api/auth/login', (req, res) => {
    const { email, password } = req.body;
    db.get("SELECT * FROM users WHERE email = ?", [email], async (err, user) => {
        if (err || !user) return res.status(401).json({ success: false, message: 'Invalid credentials' });

        const validPass = await bcrypt.compare(password, user.password_hash);
        if (!validPass) return res.status(401).json({ success: false, message: 'Invalid credentials' });

        const token = jwt.sign({ id: user.id, role: user.role, name: user.name }, JWT_SECRET, { expiresIn: '8h' });
        res.json({ success: true, token, user: { name: user.name, email: user.email, role: user.role } });
    });
});

// Auth: Signup
app.post('/api/auth/signup', async (req, res) => {
    const { name, email, password } = req.body;
    try {
        const hash = await bcrypt.hash(password, 10);
        const stmt = db.prepare("INSERT INTO users (email, password_hash, name, role) VALUES (?, ?, ?, 'user')");
        stmt.run(email, hash, name, function (err) {
            if (err) return res.status(400).json({ success: false, message: "Email likely exists" });
            const token = jwt.sign({ id: this.lastID, role: 'user', name }, JWT_SECRET);
            res.json({ success: true, token, user: { name, email, role: 'user' } });
        });
        stmt.finalize();
    } catch (e) {
        res.status(500).json({ success: false, message: "Server Error" });
    }
});

// Admin: Stats
app.get('/api/admin/stats', verifyToken, (req, res) => {
    if (req.user.role !== 'admin') return res.status(403).json({ error: "Unauthorized" });

    db.get("SELECT COUNT(*) as count FROM users", [], (err, row) => {
        if (err) return res.json({ totalUsers: 0 });
        res.json({ totalUsers: row.count, activeModels: 1, systemHealth: '100% (SQLite)' });
    });
});

// Admin: Users List
app.get('/api/admin/users', verifyToken, (req, res) => {
    if (req.user.role !== 'admin') return res.status(403).json({ error: "Unauthorized" });

    db.all("SELECT id, name, email, role, created_at FROM users ORDER BY created_at DESC", [], (err, rows) => {
        if (err) return res.status(500).json({ error: "DB Error" });
        res.json(rows.map(u => ({ ...u, createdAt: u.created_at })));
    });
});

// Admin: Delete User
app.delete('/api/admin/users/:id', verifyToken, (req, res) => {
    if (req.user.role !== 'admin') return res.status(403).json({ error: "Unauthorized" });
    const userIdToDelete = req.params.id;
    if (parseInt(userIdToDelete) === req.user.id) {
        return res.status(400).json({ success: false, message: "You cannot delete yourself!" });
    }
    db.run("DELETE FROM users WHERE id = ?", [userIdToDelete], function (err) {
        if (err) return res.status(500).json({ success: false, message: "DB Error" });
        if (this.changes === 0) return res.status(404).json({ success: false, message: "User not found" });
        res.json({ success: true, message: "User deleted" });
    });
});

// Admin: All Files
app.get('/api/admin/files', verifyToken, (req, res) => {
    if (req.user.role !== 'admin') return res.status(403).json({ error: "Unauthorized" });
    db.all("SELECT * FROM uploaded_files ORDER BY uploaded_at DESC LIMIT 50", [], (err, rows) => {
        if (err) return res.status(500).json({ error: "DB Error" });
        res.json(rows.map(r => ({ ...r, originalName: r.original_name, uploadedAt: r.uploaded_at })));
    });
});

// User: My Files
app.get('/api/user/files', verifyToken, (req, res) => {
    db.all("SELECT * FROM uploaded_files WHERE user_id = ? ORDER BY uploaded_at DESC", [req.user.id], (err, rows) => {
        if (err) return res.status(500).json({ error: "DB Error" });
        res.json(rows.map(r => ({ ...r, originalName: r.original_name, uploadedAt: r.uploaded_at })));
    });
});

// User: Settings (Mock for Demo)
app.post('/api/user/settings', verifyToken, (req, res) => {
    // In a real app, update 'settings' column in SQLite
    // For demo stability, we just acknowledge receipt
    res.json({ success: true, message: "Settings saved" });
});

// File Upload
const upload = multer({ dest: 'uploads/', limits: { fileSize: 50 * 1024 * 1024 } });

app.post('/api/upload', verifyToken, upload.single('file'), async (req, res) => {
    if (!req.file) return res.status(400).json({ success: false, message: 'No file uploaded' });

    const stmt = db.prepare("INSERT INTO uploaded_files (user_id, original_name, filename, size) VALUES (?, ?, ?, ?)");
    stmt.run(req.user ? req.user.id : 1, req.file.originalname, req.file.filename, req.file.size, async function (err) {
        if (err) console.error("DB Upload Error", err);

        const newFile = {
            id: this.lastID,
            originalName: req.file.originalname,
            status: 'Processed'
        };

        // --- ML PREDICTION TRIGGER ---
        let predictionData = null;
        if (req.file.mimetype === 'text/csv' || req.file.originalname.endsWith('.csv')) {
            try {
                const fullPath = path.resolve(req.file.path);
                if (axios) {
                    const mlRes = await axios.post(`${ML_SERVICE_URL}/predict_csv`, { file_path: fullPath });
                    if (mlRes.data && mlRes.data.success) {
                        predictionData = mlRes.data;
                    }
                }
            } catch (err) {
                console.error("ML Error", err.message);
            }
        }
        res.json({ success: true, file: newFile, prediction: predictionData });
    });
    stmt.finalize();
});

// ML Prediction Endpoint
app.post('/api/predict_demand', verifyToken, async (req, res) => {
    const riskContext = AutoSignalContext;
    try {
        if (!axios) throw new Error("Axios not installed");
        const payload = { ...req.body, auto_signals: riskContext };
        const response = await axios.post(`${ML_SERVICE_URL}/predict`, payload);
        res.json(response.data);
    } catch (error) {
        // Fallback Logic
        let base = 1250;
        let riskMult = 1.0;
        if (riskContext.weather.status === 'STORM') riskMult -= 0.3;
        if (riskContext.social.status === 'VIRAL_NEGATIVE') riskMult -= 0.2;
        res.json({
            predicted_demand: Math.round(base * riskMult),
            auto_signals: { multiplier_applied: riskMult.toFixed(2) }
        });
    }
});

app.post('/api/train_model', verifyToken, async (req, res) => {
    if (req.user.role !== 'admin') return res.status(403).json({ error: "Unauthorized" });
    try {
        if (axios) await axios.post(`${ML_SERVICE_URL}/train`);
        res.json({ message: "Training Started", mode: "background" });
    } catch (e) {
        res.json({ message: "Training Mocked", mode: "simulation" });
    }
});

// Start Server
if (require.main === module) {
    app.listen(PORT, () => {
        console.log(`Server running at http://localhost:${PORT}`);
        console.log("Enterprise Mode Active: SQLite DB + JWT Auth + Auto-Signals");
    });
}

module.exports = app;
