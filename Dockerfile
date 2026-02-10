# 1. Use an official lightweight Python image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy dependencies first (for better caching)
COPY requirements.txt .

# 4. Install dependencies
# We add --no-cache-dir to keep the image small
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the application code
# This copies: src/, app/, data/, models/, and train.py
COPY . .

# 6. Expose the port the app runs on
EXPOSE 8000

# 7. Define the command to run the app
# host 0.0.0.0 is crucial for Docker networking
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]