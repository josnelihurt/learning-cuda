#!/bin/bash

bazel run //:gazelle-update-repos
bazel build //webserver/cmd/server:server