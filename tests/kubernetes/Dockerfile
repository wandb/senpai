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
COPY scripts/senpai-browser-smoke-test.py /usr/local/bin/senpai-browser-smoke-test
COPY scripts/senpai-container-health.sh /usr/local/bin/senpai-container-health
COPY scripts/senpai-run-controller.sh /usr/local/bin/senpai-run-controller
COPY scripts/senpai-metadata-egress-guard.py /usr/local/bin/senpai-metadata-egress-guard
COPY senpai_agent/repair_executor.py /usr/local/bin/senpai-repair-executor
COPY k8s/senpai-role-shell /usr/local/bin/senpai-role-shell

# The base image carries dependencies, while every canary invocation must run
# the source under test even through Python's isolated (-I) entrypoints.
RUN site_packages=$(/opt/senpai-venv/bin/python -I -c \
      'import site; print(site.getsitepackages()[0])') && \
    rm -rf "$site_packages/senpai_agent" && \
    cp -a /opt/senpai/senpai_agent "$site_packages/senpai_agent" && \
    chown -R 0:0 "$site_packages/senpai_agent" && \
    chmod -R go-w "$site_packages/senpai_agent" && \
    chmod 0755 \
      /usr/local/bin/senpai-browser-smoke-test \
      /usr/local/bin/senpai-container-health \
      /usr/local/bin/senpai-run-controller \
      /usr/local/bin/senpai-metadata-egress-guard \
      /usr/local/bin/senpai-repair-executor \
      /usr/local/bin/senpai-role-shell

ENV PYTHONPATH=/opt/senpai
USER 10001:10001
WORKDIR /workspaces
