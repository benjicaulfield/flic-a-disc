FROM node:20-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy source code
COPY . .

# Copy .env.example as template
COPY .env.example .env.local

EXPOSE 5173

# Default command (overridden in docker-compose)
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
