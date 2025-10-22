FROM alpine:3.19

RUN apk add --no-cache \
    bash \
    ripgrep \
    git \
    perl

COPY scripts/linters/language-check.sh /usr/local/bin/language-check
RUN chmod +x /usr/local/bin/language-check

WORKDIR /workspace

ENTRYPOINT ["/usr/local/bin/language-check"]

