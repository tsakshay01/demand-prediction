# Base Image: Node.js 20 (LTS)
FROM node:20-slim

WORKDIR /app

# Install Dependencies
COPY package*.json ./
RUN npm install --production

# Copy App Code
COPY . .

# Expose Port
EXPOSE 3000

# Environment Defaults
ENV PORT=3000
# Important: In Docker Compose, the host is the *service name*, not localhost
ENV ML_SERVICE_URL=http://ml-service:5000

# Start Server
CMD ["node", "server.js"]
