#!/bin/bash

set -e

cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

if [ ! -f "proto/gen/config_service.pb.go" ]; then
    docker run --rm -v $(pwd):/workspace -u $(id -u):$(id -g) cuda-learning-bufgen:latest generate
else
    PROTO_MOD=$(stat -c %Y proto/config_service.proto 2>/dev/null || stat -f %m proto/config_service.proto)
    GEN_MOD=$(stat -c %Y proto/gen/config_service.pb.go 2>/dev/null || stat -f %m proto/gen/config_service.pb.go)
    
    [ "$PROTO_MOD" -gt "$GEN_MOD" ] && \
        docker run --rm -v $(pwd):/workspace -u $(id -u):$(id -g) cuda-learning-bufgen:latest generate
fi

grep -q "ListInputs" proto/gen/config_service.pb.go || {
    echo "ListInputs not found in generated proto"
    exit 1
}

grep -q "type InputSource struct" proto/gen/config_service.pb.go || {
    echo "InputSource not found in generated proto"
    exit 1
}

