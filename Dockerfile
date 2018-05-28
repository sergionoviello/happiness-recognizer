FROM ermaker/keras

RUN conda install -y \
    jupyter \
    matplotlib \
    seaborn

RUN pip install flask \
    tensorflow --upgrade

RUN pip install pillow

VOLUME /app
WORKDIR /app

EXPOSE 5000

CMD "./start.sh"