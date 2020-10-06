FROM python:3.7

COPY * /tmp/

RUN conda install python=3.8.3
RUN conda install faiss-cpu=1.5.1 -c pytorch -y

WORKDIR /tmp

CMD ["streamlit", "run", "steamlit_app.py"]
