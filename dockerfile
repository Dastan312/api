FROM python:3.10

WORKDIR /API

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]



# FROM python:3.10

# WORKDIR /app

# COPY src/main.py /app/main.py
# COPY src/requirements.txt /app/requirements.txt

# RUN pip install --no-cache-dir --upgrade -r requirements.txt

# CMD ["python", "main.py"]
