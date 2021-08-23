# SPDX-License-Identifier: Apache-2.0

FROM nvcr.io/nvidia/clara-train-sdk:v4.0

ARG USER=adev
ARG UID=1000
ARG GID=1000

RUN apt-get update && apt-get upgrade -y
RUN pip install --upgrade pip
RUN apt update
RUN apt install -y build-essential

#### jupyterlab-nvdashboard dependences (nodesj and npm installation)
RUN curl -sL https://deb.nodesource.com/setup_12.x | bash -
RUN apt install -y nodejs
##RUN mv /usr/bin/python /usr/bin/python2 # dir doesn't exist !
RUN ln -s $(which python3) /usr/bin/python

####
#Prerequisites : JupyterLab 1.0 bokeh pynvml
RUN pip install pynvml
RUN pip install jupyter-server-proxy
####

### bokeh dependences
RUN pip install Jinja2 \
    && pip install numpy \
    && pip install packaging \
    && pip install pillow \
    && pip install python-dateutil \
    && pip install six  \
    && pip install tornado
RUN pip install bokeh
RUN pip install jupyterlab
RUN pip install jupyterlab-nvdashboard
# RUN jupyter labextension install jupyterlab-nvdashboard
# RUN pip install git+https://github.com/cliffwoolley/jupyter_tensorboard.git
RUN pip install git+https://github.com/chaoleili/jupyterlab_tensorboard.git

RUN pip install sanic
RUN apt-get install -y dcm2niix

# For workshops copy content inside docker image
#COPY ./ /claraDevDay/


# Set the home directory to our user's home.
ENV USER=$USER
ENV HOME="/home/$USER"

RUN echo "Create $USER account" &&\
    # Create the home directory for the new $USER
    mkdir -p $HOME &&\
    # Create an $USER so our program doesn't run as root.
    groupadd -r -g $GID $USER &&\
    useradd -r -g $USER -G sudo -u $UID -d $HOME -s /sbin/nologin -c "Docker image user" $USER &&\
    # Set root user no password
    mkdir -p /etc/sudoers.d &&\
    echo "$USER ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USER && \
    chmod 0440 /etc/sudoers.d/$USER

COPY .bashrc $HOME/
# Chown all the files to the $USER
RUN chown -R $USER:$USER $HOME


WORKDIR /claraDevDay/
EXPOSE 8888
CMD ["/usr/local/bin/jupyter", "lab", "/claraDevDay", "--ip", "0.0.0.0", "--port", "8888" ,"--allow-root","--no-browser"]
