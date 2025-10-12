#!/bin/bash
set -e
bazel clean --expunge
go clean -cache -modcache -testcache
rm -f webserver/cmd/server/server
rm -rf proto/gen/*
