FROM python:3.7
MAINTAINER Micah Melling, micahmelling@gmail.com
RUN mkdir /app
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
ENTRYPOINT ["python3"]
