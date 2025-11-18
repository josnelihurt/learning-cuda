def _googleapis_imports_repo_impl(ctx):
    ctx.file("BUILD.bazel", "")
    ctx.file(
        "imports.bzl",
        """load("@com_github_grpc_grpc//bazel:cc_grpc_library.bzl", grpc_cc_library = "cc_grpc_library")

def proto_library_with_info(**kwargs):
    native.proto_library(**kwargs)

def moved_proto_library(**kwargs):
    native.proto_library(**kwargs)

def cc_proto_library(**kwargs):
    native.cc_proto_library(**kwargs)

def cc_grpc_library(**kwargs):
    grpc_cc_library(**kwargs)

def java_proto_library(**kwargs):
    pass

def java_grpc_library(**kwargs):
    pass

def java_gapic_library(**kwargs):
    pass

def java_gapic_test(**kwargs):
    pass

def java_gapic_assembly_gradle_pkg(**kwargs):
    pass

def py_proto_library(**kwargs):
    pass

def py_grpc_library(**kwargs):
    pass

def py_gapic_library(**kwargs):
    pass

def py_gapic_assembly_pkg(**kwargs):
    pass

def go_proto_library(**kwargs):
    pass

def go_library(**kwargs):
    pass

def go_test(**kwargs):
    pass

def go_gapic_library(**kwargs):
    pass

def go_gapic_assembly_pkg(**kwargs):
    pass

def php_proto_library(**kwargs):
    pass

def php_grpc_library(**kwargs):
    pass

def php_gapic_library(**kwargs):
    pass

def php_gapic_assembly_pkg(**kwargs):
    pass

def nodejs_proto_library(**kwargs):
    pass

def nodejs_grpc_library(**kwargs):
    pass

def nodejs_gapic_library(**kwargs):
    pass

def nodejs_gapic_assembly_pkg(**kwargs):
    pass

def ruby_proto_library(**kwargs):
    pass

def ruby_grpc_library(**kwargs):
    pass

def ruby_gapic_library(**kwargs):
    pass

def ruby_gapic_assembly_pkg(**kwargs):
    pass

def csharp_proto_library(**kwargs):
    pass

def csharp_grpc_library(**kwargs):
    pass

def csharp_gapic_library(**kwargs):
    pass

def csharp_gapic_assembly_pkg(**kwargs):
    pass

def proto_grpc_library(**kwargs):
    grpc_cc_library(**kwargs)
""",
    )

googleapis_imports_repo = repository_rule(
    implementation = _googleapis_imports_repo_impl,
)

