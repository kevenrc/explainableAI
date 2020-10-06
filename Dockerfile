FROM python:3.7

COPY streamlit_app.py /tmp/
COPY requirementsltxt /tmp/
COPY .streamlit /tmp/.streamlit/

RUN conda install python=3.8.3
RUN conda install faiss-cpu=1.5.1 -c pytorch -y

WORKDIR /tmp

RUN apt-get update && apt-get install -y g++
RUN pip install -r requirement.txt

CMD ["streamlit", "run", "steamlit_app.py"]
