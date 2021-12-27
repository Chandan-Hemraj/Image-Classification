FROM python:3

ADD app.py /
ADD model.hdf5 /

COPY requirements.txt ./
RUN pip3 install -r requirements.txt

ENTRYPOINT [ "python", "app.py"]