FROM python:3.10.4

# Create the working directory
RUN set -ex && mkdir /translation
WORKDIR /translation

# Install Python dependencies
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Copy the relevant directories
COPY . ./

# Run the web server
CMD ["python", "app.py"]