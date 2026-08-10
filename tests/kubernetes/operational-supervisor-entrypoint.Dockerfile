ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# This image exists only to observe the production entrypoint's final exec.
# No production flag or branch changes runtime behavior.
USER 0:0
COPY operational-supervisor-controller-stub.sh /usr/local/bin/senpai-run-controller
RUN chmod 0755 /usr/local/bin/senpai-run-controller
USER 10001:10001
