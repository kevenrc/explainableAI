FROM python:3.7

COPY * /tmp/

WORKDIR /tmp

RUN apt-get update && apt-get install -y g++
RUN pip3 install -r requirements.txt --cache-dir=/tmp/

CMD ["streamlit", "run", "steamlit_app.py"]
