# DemandAI - Project Explanation & Placement Guide

## 1. Project Overview
**Name**: DemandAI (Enterprise Supply Chain Demand Prediction System)
**Tagline**: "Bridging the gap between historical data and future trends using Multimodal AI."

**Core Function**: 
DemandAI is a full-stack web application that predicts future product demand. Unlike traditional systems that only look at past sales numbers (time-series), this system is **Multimodal**—it effectively "looks" at the product image and "reads" the product description, combining these insights with sales history to generate highly accurate predictions.

### Key Features
- **Multimodal Analysis**: Combines Text (NLP), Image (Computer Vision), and Time-series data.
- **Enterprise Architecture**: Decoupled Microservices (Node.js + Python).
- **Secure Access**: Role-Based Access Control (RBAC) with JWT and Bcrypt.
- **Real-time & Batch Processing**: Supports single file uploads and bulk CSV processing.

---

## 2. Tech Stack (The "What")

| Component | Technology | Why we chose it? |
|-----------|------------|------------------|
| **Frontend** | HTML5, CSS3, Vanilla JS | Lightweight, fast performance, no heavy framework overhead for simple dashboards. |
| **Backend** | **Node.js + Express** | Non-blocking I/O is perfect for handling concurrent user requests and file uploads. |
| **ML Engine** | **Python (Flask)** | Python is the industry standard for AI. Flask provides a lightweight API wrapper. |
| **Database** | **SQLite** | Zero-configuration, serverless SQL engine. Perfect for portable deployment and demos. |
| **Deep Learning** | **TensorFlow / Keras** | Robust ecosystem for building complex custom neural network architectures. |
| **NLP** | **HuggingFace Transformers (DistilBERT)** | State-of-the-art language understanding for product descriptions. |
| **Computer Vision** | **MobileNetV2** | Efficient, pre-trained image recognition model that runs fast on standard hardware. |

---

## 3. High-Level Architecture (The "How")

The system follows a **Microservices-based Architecture** patterns:
1.  **Orchestrator (Node.js)**: Handles client communication, authentication, and logic.
2.  **Worker (Python ML Service)**: Dedicated computational unit for heavy AI inference.

```mermaid
graph TD
    User[User (Browser)] -->|Uploads CSV/Image| Node[Node.js Server (Port 3000)]
    Node -->|Auth Check| DB[(SQLite Database)]
    Node -->|Saves File| Storage[File System]
    Node -->|POST /predict| Python[Python Flask ML Service (Port 5000)]
    
    subgraph "ML Service Internals"
        Python -->|Text| BERT[DistilBERT (NLP)]
        Python -->|Image| CNN[MobileNetV2 (Vision)]
        Python -->|History| Transformer[Transformer/Attention (Time Series)]
        BERT & CNN & Transformer --> Fusion[Fusion Layer]
        Fusion --> Prediction[Final Demand Value]
    end
    
    Python -- Returns JSON --> Node
    Node -- Displays Dashboard --> User
```

1.  **Frontend**: User logs in and uploads a dataset.
2.  **Node.js**:
    *   Validates the JWT Token.
    *   Streams the file to the `uploads/` folder.
    *   Records metadata in SQLite.
    *   Sends an HTTP POST request to the Python Service with the file path.
3.  **Python Flask**:
    *   Reads the CSV from the disk.
    *   **Pre-processing**: Normalizes numbers (0-1 scale), tokenizes text, resizes images.
    *   **Inference**: Passes data through the loaded Neural Network.
    *   Returns the prediction.
4.  **Node.js**: Relays the result to the Frontend for visualization (Chart.js).

---

## 4. The ML Model: Why & How?
**Why this model?**
Standard demand forecasting uses ARIMA or simple LSTMs (Recurrent Neural Networks) on just numbers. However, *what* a product is (its visual appeal, its description) strongly influences demand. We built a **Hybrid Multimodal Network** to capture this.

### Architecture Breakdown (See `ml_service/model.py`)
1.  **Text Branch (DistilBERT)**:
    *   **What**: Processes product descriptions (e.g., "Vintage blue denim jacket").
    *   **Why**: Helps group similar items even if they have different IDs. "Red Dress" and "Crimson Gown" are semantically close.
2.  **Image Branch (MobileNetV2)**:
    *   **What**: Extracts visual features from product images.
    *   **Why**: Visual aesthetics drive fashion/retail demand. MobileNetV2 is used because it's fast (low latency).
3.  **Time-Series Branch (Transformer)**:
    *   **What**: An Attention-based Transformer block (upgraded from standard LSTM).
    *   **Why**: Transformers handle long-term dependencies (e.g., sales from 12 months ago) better than RNNs and accommodate parallel training.

---

## 5. Challenges & Solutions (The "Star" Method)

**Challenge 1: System Integration (The "Works on my machine" problem)**
*   **Situation**: Connecting a Node.js server to a Python script is tricky.
*   **Task**: Make them communicate reliably.
*   **Action**: We implemented a RESTful API architecture. Instead of spawning fragile child processes ( `spawn('python')` ), we run the Python script as a standalone Flask server (`app.py`).
*   **Result**: The Node.js server simply sends HTTP requests (`axios.post`), making the system decoupled and scalable. If the ML model crashes, the Web Server stays alive.

**Challenge 2: Data Scaling & Normalization**
*   **Situation**: The model initially refused to learn (constant flat predictions).
*   **Root Cause**: Neural Networks expect inputs between 0 and 1. Our sales data was in the millions (e.g., 2,000,000). This caused "exploding gradients."
*   **Action**: Implemented robust `MinMaxScaler` logic. We scale inputs down (Input / Max_Value) before inference and de-scale outputs (Prediction * Max_Value) before displaying them.
*   **Result**: The model could finally learn patterns and converge.

**Challenge 3: Handling Missing Data**
*   **Situation**: In real-world testing, users often uploaded CSVs without image paths or descriptions.
*   **Action**: Built a "Zero-Tolerance" fallback system in `model.py`. If an image is missing, we inject a "Zero Tensor" (a black image placeholder) rather than crashing. If text is missing, we use a generic "unknown" token.
*   **Result**: The system is crash-proof and handles messy real-world data gracefully.

---

## 6. Possible Interview Questions & Answers

**Q: Why did you use SQLite instead of MySQL/MongoDB?**
**A:** For this specific deployment, we prioritized portability. SQLite is serverless and file-based (`demand_ai.db`), allowing the entire project to be zipped and run on any machine (like this demo laptop) without complex database installation. For production, we would switch to PostgreSQL or MongoDB Atlas, and our Code is modular (`database.js` wrapper) allows this swap easily.

**Q: Why separate the ML service? Why not run it inside Node?**
**A:** Node.js is single-threaded and optimized for I/O. Meaning it's great at moving data around but terrible at heavy math. If we ran the model inside Node, a single prediction would freeze the entire website for all other users. By offloading it to Python (Flask), Node remains fast and responsive.

**Q: Explain the "Transformer" part of your model.**
**A:** Traditional models read data step-by-step (Monday, then Tuesday...). This is slow and forgets long-term history. Our Transformer uses a mathematical mechanism called "Self-Attention" to look at the entire 30-day history *at once*, identifying distinct patterns (like "Sales spike every Friday") much more effectively.

**Q: How do you handle security?**
**A:** 
1.  **Authentication**: We use JWT (JSON Web Tokens). When a user logs in, they get a signed token which must be sent in the Header of every request.
2.  **Passwords**: We never store plain text. We hash passwords using `bcrypt` with salt rounds.
3.  **API Protection**: We use `helmet` for header security and `express-rate-limit` to prevent DDoS attacks.

---

## 7. Resume Entry

Here is the exact format to copy-paste into your Resume:

**DemandAI – Enterprise Multimodal Supply Chain Prediction Information System**
**Team Size:** 1 | **Role:** Full Stack AI Developer  
**Duration:** 1 Month  
**Technologies Used:** Python (Flask, TensorFlow/Keras), Node.js (Express), SQLite, HuggingFace Transformers (DistilBERT), MobileNetV2

*   Developed a multimodal AI system capable of predicting future product demand by analyzing product images (Computer Vision), descriptions (NLP), and historical sales data (Time-Series).
*   Implemented a microservices architecture with a Node.js backend for secure data management and a Python Flask service for real-time deep learning inference.
*   Designed a robust data pipeline using **MinMaxScaler** normalization to handle variable demand scales from 100 to 2,000,000 units.
*   Built a secure REST API with **JWT Authentication** and **RBAC** to protect enterprise data and enable role-specific dashboards.
