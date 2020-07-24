# The first instruction is what image we want to base our container on
# We Use an official Python runtime as a parent image
FROM python:2.7

RUN apt-get update

RUN apt-get install rabbitmq-server -y

# The enviroment variable ensures that the python output is set straight
# to the terminal with out buffering it first
ENV PYTHONUNBUFFERED 1

# create root directory for our project in the container
RUN apt-get install python3-tk -y
RUN pip install --upgrade pip
RUN mkdir /dltk-vision-python

# Set the working directory to /cloud-chatbot
WORKDIR /dltk-vision-python

# Copy the current directory contents into the container at /cloud-chatbot
ADD . /dltk-vision-python/

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt
