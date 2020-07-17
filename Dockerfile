FROM python:3.6

WORKDIR /var/lib/p300

ADD ./requirements.txt .
RUN pip install -r ./requirements.txt

ADD ./download.sh .
RUN ./download.sh