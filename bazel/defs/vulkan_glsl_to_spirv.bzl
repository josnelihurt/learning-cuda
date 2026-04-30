load("//bazel/defs:spirv_embed.bzl", "spirv_shader_embed_cpp")

def vulkan_kernel_embedded(name, shader):
    """Compile a GLSL compute shader to SPIR-V and embed it for linking.

    Mirrors ``opencl_kernel_embedded`` in naming convention. Use ``name`` as the stem shared
    by outputs and C symbols (no extension). Typical layout: ``shader = name + ".comp"``.

    Generated targets:

    - ``:{name}_spirv``       — ``glslc`` genrule → ``{name}.spv``.
    - ``:{name}_spv_embed``   — ``xxd -i`` of ``{name}.spv`` → ``{name}_spv[]``, ``{name}_spv_len``.
    - ``:{name}_blob_h``      — ``{name}_blob.h`` with ``struct {name}_blob`` (``spirv()`` /
      ``spirv_size_bytes()``).

    Typical pattern: one ``cc_library`` with ``srcs`` listing your ``.cpp`` and
    ``\":{name}_spv_embed\"``, and ``hdrs`` listing ``\":{name}_blob_h\"``.  Include as
    ``#include \"src/cpp_accelerator/adapters/compute/vulkan/{name}_blob.h\"``.

    Requires ``glslc`` on ``PATH`` during the build (Vulkan SDK or distro package).
    """
    spirv_shader_embed_cpp(
        name = name,
        shader = shader,
    )
