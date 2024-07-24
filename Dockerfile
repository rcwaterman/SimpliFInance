FROM python:3.11.4
RUN apt-get update && apt-get install -y \
    curl \
    net-tools \
    iputils-ping
EXPOSE 8080
EXPOSE 7860
EXPOSE 80
EXPOSE 443
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app
COPY --chown=user . $HOME/app
COPY ./requirements.txt ~/app/requirements.txt
RUN pip install -r requirements.txt
COPY . .
CMD ["chainlit", "run", "app.py", "--port", "7860"]