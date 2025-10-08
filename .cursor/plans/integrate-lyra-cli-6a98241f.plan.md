<!-- 6a98241f-d395-42b5-be02-fa5bccbdb3aa 3221b269-b953-48fd-8148-6a30b15325a0 -->
# Integrate Lyra with Clean Architecture Config Layer

## 1. Integrar Lyra con Bazel

Dado que Lyra es header-only y no existe un módulo oficial de Bazel, usaremos `http_archive`:

- Actualizar `MODULE.bazel`:
  - Agregar `http_archive` para descargar Lyra desde GitHub (versión 1.6.1)
  - URL: `https://github.com/bfgroup/Lyra/archive/refs/tags/1.6.1.tar.gz`

- Crear `third_party/lyra/BUILD`:
  - Definir `cc_library` header-only que exponga `lyra.hpp`
  - Visibilidad pública

## 2. Crear capa de configuración (Clean Architecture)

Estructura de directorios:

```
config/
  ├── models/          # jrb::config::models
  │   └── program_config.h
  └── config_manager.h/.cpp  # jrb::config
```

### 2.1 DTO/Modelo de configuración

Crear `config/models/program_config.h`:

- Struct `ProgramConfig` con:
  - `std::string input_image_path`
  - `std::string output_image_path`
  - `enum class ProgramType { Simple, Grayscale }`
  - `ProgramType program_type`

### 2.2 ConfigManager

Crear `config/config_manager.h`:

- Clase `ConfigManager` que:
  - Constructor: `ConfigManager(int argc, const char** argv)`
  - Procesa argumentos usando Lyra
  - Método público: `ProgramConfig get_config() const`
  - Validación de argumentos
  - Generación automática de ayuda

Crear `config/config_manager.cpp`:

- Implementar parsing con Lyra:
  - `-i, --input`: ruta de imagen de entrada (default: "data/lena.png")
  - `-o, --output`: ruta de imagen de salida (default: "data/output.png")
  - `-t, --type`: tipo de programa: "simple" o "grayscale" (default: "grayscale")
  - `--help`: mostrar ayuda

### 2.3 BUILD file

Crear `config/BUILD`:

- `cc_library` para `config` que incluya:
  - `models/program_config.h` (header)
  - `config_manager.h/.cpp`
  - Dependencia: `//third_party/lyra:lyra`

## 3. Actualizar main.cpp

Modificar `cmd/hello_cuda/main.cpp`:

- Incluir `config/config_manager.h`
- Crear `ConfigManager` con argc/argv
- Obtener `ProgramConfig`
- Implementar switch case basado en `program_type`:
  - `ProgramType::Simple`: ejecutar `launch_hello_kernel()`
  - `ProgramType::Grayscale`: ejecutar pipeline de grayscale con paths del config
- Manejo de errores de configuración

## 4. Actualizar dependencias

Modificar `cmd/hello_cuda/BUILD`:

- Agregar dependencia: `//config:config`

## Estructura final de namespaces

- `jrb::config::models` - DTOs y modelos de configuración
- `jrb::config` - ConfigManager y lógica de configuración
- Main usa config layer sin conocer detalles de implementación

### To-dos

- [ ] Integrar Lyra: actualizar MODULE.bazel con http_archive y crear third_party/lyra/BUILD
- [ ] Crear DTO: config/models/program_config.h con ProgramConfig struct y ProgramType enum
- [ ] Crear ConfigManager: config/config_manager.h/.cpp con parsing de Lyra y validación
- [ ] Crear config/BUILD con dependencias apropiadas
- [ ] Actualizar cmd/hello_cuda/main.cpp con switch case y uso de ConfigManager
- [ ] Compilar y probar con diferentes argumentos CLI