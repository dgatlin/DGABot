# Use an official Python runtime as a parent image
FROM python:3.6.8-slim

# Set the working directory to /dgabot
WORKDIR /dgabot

# Copy the current directory contents into the container at /app
COPY . /dgabot

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run app.py when the container launches
CMD ["python", "app.py"]