# Use a lightweight base image
FROM ubuntu:20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH /opt/conda/bin:$PATH

# Install dependencies
RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 && \
    apt-get clean

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy

# Create and activate conda environment, then install PRGpred
COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && \
    conda clean -a

# Set the default environment
ENV CONDA_DEFAULT_ENV=PRGpred
ENV PATH /opt/conda/envs/$CONDA_DEFAULT_ENV/bin:$PATH

# Copy PRGpred source code and install
COPY . /PRGpred
WORKDIR /PRGpred
RUN pip install .

# Command to run PRGpred
CMD ["PRGminer", "-i", "input.fasta", "-od", "results", "-l", "Phase1"]