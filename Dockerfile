FROM nvcr.io/nvidia/pytorch:21.08-py3 
#yilei
RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python -m pip install --user -U pip

COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
RUN python -m pip install --user -r requirements.txt

COPY --chown=algorithm:algorithm /sample_data /
COPY --chown=algorithm:algorithm /model /opt/algorithm/my_model
#yilei

COPY --chown=algorithm:algorithm process.py /opt/algorithm/
COPY --chown=algorithm:algorithm settings.py /opt/algorithm/
COPY --chown=algorithm:algorithm grandchallenges/ /opt/algorithm/grandchallenges
# COPY --chown=algorithm:algorithm isles/ /opt/algorithm/isles
ENTRYPOINT python -m process $0 $@
