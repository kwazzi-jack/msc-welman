FROM ubuntu:20.04

# Add KERN-suite PPA
RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository -sy ppa:kernsuite/kern-8
RUN apt-add-repository -y multiverse
RUN apt-add-repository -y restricted
RUN apt-get update

# Install system dependencies
RUN apt-get install -y \
	wget \
	git \
	python3.8 \
	python3.8-distutils \
	python3.8-apt \
	python3.8-dev \
	build-essential \
	python3-pip \
	python3-venv \
	python3-casacore \
	casalite

# Create the simulation directory
RUN mkdir -p /home/simulation/main
WORKDIR /home/simulation

# Copy the Jupyter Notebook files and other code to the container
COPY simulation.ipynb main/
COPY meerkat-src-100-config.yml main/
COPY source main/source

# Copy the requirements.txt file
COPY requirements.txt .

# Create a virtual environment in the desired location
RUN python3.8 -m venv /home/simulation/venv

# Activate the virtual environment
ENV PATH="/home/simulation/venv/bin:$PATH"

# Install required libraries
RUN pip install --no-cache-dir -r requirements.txt

# Expose the Jupyter Notebook port
EXPOSE 8888

# Start Jupyter Notebook when the container launches
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

