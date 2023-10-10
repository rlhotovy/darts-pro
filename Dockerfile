# syntax=docker/dockerfile:1.2
FROM python:3.9.6-buster

# For persisting bash history
RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" && echo $SNIPPET >> "/root/.bashrc"

RUN python3 -m pip install pipx==0.16.4 && \
    python3 -m pipx ensurepath

# Poetry
ENV POETRY_VERSION 1.5.0
ENV POETRY_HOME /root/.poetry
# curl https://install.python-poetry.org - this causes a Segmentation Fault! Madness
RUN wget -O - -o /dev/null https://install.python-poetry.org | python3
ENV PATH /root/.poetry/bin:${PATH}

COPY poetry.* ./
COPY pyproject.toml ./
ARG DEV=--no-dev
RUN poetry install ${DEV}
ADD . .

ENTRYPOINT ["/usr/local/bin/python", "main.py"]
CMD [ "--", "--help" ]
