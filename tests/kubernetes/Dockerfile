ARG BASE_IMAGE
FROM ${BASE_IMAGE}

USER 0:0
WORKDIR /opt/senpai

# Deliberately enumerate the runtime surface. In particular, never copy the
# checkout's .git directory, GitHub credential helper, or a developer .env.
COPY --chown=10001:10001 k8s/ /opt/senpai/k8s/
COPY --chown=10001:10001 senpai/ /opt/senpai/senpai/
COPY --chown=10001:10001 senpai_agent/ /opt/senpai/senpai_agent/
COPY --chown=10001:10001 system_instructions/ /opt/senpai/system_instructions/
COPY --chown=10001:10001 senpai.yaml /opt/senpai/senpai.yaml
COPY --chown=10001:10001 tests/kubernetes/canary.py /opt/senpai/tests/kubernetes/canary.py

ENV PYTHONPATH=/opt/senpai
USER 10001:10001
WORKDIR /workspaces
