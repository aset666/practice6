# use official lightweight Python image
FROM python:3.11-slim
# set working directory inside the container
WORKDIR /app
# copy dependency list first (layer caching)
COPY requirements.txt .
# install dependencies
RUN pip install --no-cache-dir -r requirements.txt
# copy the rest of the project files
COPY . .
# train the model so model.joblib is available inside the image
RUN python train.py
# expose port 8000
EXPOSE 8000
# run the FastAPI app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
