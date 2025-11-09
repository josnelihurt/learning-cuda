Titulo: Implementation for video playback from files using BDD, features, backend front-end, testing and e2e

## Prerequisites - Read These First

1. `.prompts/development/add-dynamic-feature-end-to-end.md` - Development workflow and file structure
2. `.prompts/development/golang-testing-best-practices.md` - Go testing patterns (AAA, testify/mock, sut)
3. `.prompts/development/backlog-and-commit-workflow.md` - Documentation and commit message process
4. `scripts/` - Use existing scripts: start-dev.sh, run-linters.sh, githooks/*

## Context:
Actualmente el sistema permite agregar fuentes input-source para enviarlas al backend y aplicar filtros sobre ellas, ahora se requiere agregar un nuevo tipo de fuente de video que se llama video, para leer archivos almacenados en el servidor hacer playback y frame a frame aplicar el filtro seleccionado (si no hay filtro no se aplica nada), luego de procesar se envia al web-socket correspondiente y frame a frame se visualiza su ejecucion, una vez el video termina se inicia desde su inicio, infinitamente hasta remover el input desde el front. Existe la funcionalidad de subir imagenes extiende esta funcionalidad para subir videos, agrega un nuevo endpoint al servicio correspondiente y aplica el mismo patron que el aplicado al de imagenes. Se requiere poder implmentar la funcionalidad de agregar la nueva fuente de video y cambiar el video similar a como sucede con las imagenes, en este caso la previsualizacion seria una imagen estatica para cada video en el modal de seleecion. La funcionaldiad de discovery de videos en disco se debe replicar como la de imagenes. El video de default debe ser seleccionable como el la imagen por defecto.

## Resultado esperado:
- BDD Implementado
- Proto files
- Cambios en golang
- Cambios en front-end
- Pruebas unitarias
    - golang
    - front-end
- Pruebas e2e corriendo en un navegador
- Pruebas manuales usando mcp integration con chrome
    - Ajuste de pruebas e2e basado en la exploracion con chrome

# Pasos previos a las pruebas.
Busca en internet un video de ejemplo y descargalo en la carpeta data/video para las pruebas, usa uno timpo en procesamiento de video.


## Execution Steps

### 1. Investigate Codebase

Search for similar features:
- `codebase_search`: "How is [similar feature] implemented?"
- Read: `integration/tests/acceptance/features/*.feature`
- Read: `proto/*.proto`
- Read: `webserver/pkg/interfaces/connectrpc/*_handler.go`

### 2. BDD Feature (Define Requirements First)

Create `integration/tests/acceptance/features/[feature_name].feature`:
- Scenarios: success, validation errors, edge cases
- Follow existing feature file patterns

Update BDD steps:
- `steps/bdd_context.go`: Add client, When*/Then* methods
- `steps/when_steps.go`: Wrappers + register
- `steps/then_steps.go`: Wrappers + register

### 3. Proto Definition

If new service:
1. Create `proto/[service]_service.proto`
2. Move shared types to `proto/common.proto` (no circular deps)
3. Import only `common.proto` 
4. Remove RPCs from old service
5. Generate: `docker run --rm -v $(pwd):/workspace -u $(id -u):$(id -g) cuda-learning-bufgen:latest generate`

Critical: Never import service protos from other service protos.

### 4. Backend (Clean Architecture Order)

**Domain**: Update `webserver/pkg/domain/[entity]_repository.go`

**Infrastructure**: 
- Implement in `webserver/pkg/infrastructure/[type]/[repository].go`
- Create tests with table-driven pattern

**Application**:
- Create `webserver/pkg/application/[feature]_use_case.go`
- Create tests following golang-testing-best-practices.md
- Use typed errors, mock.Mock, assertResult functions

**Interface**:
- Create `webserver/pkg/interfaces/connectrpc/[service]_handler.go`
- Add Register[Service]Service in `server.go`

**DI**: Wire in `container.go`, `app.go`, `cmd/server/main.go`

### 5. Frontend

**Service**: Create `webserver/web/src/services/[service]-service.ts`
- Use telemetryService.createSpan (not startSpan)
- Add null-safety: `span?.setAttribute`, `span?.end()`

**Component**: Create `webserver/web/src/components/[component].ts`
- Use Lit (@customElement, @state)
- Add data-testid attributes
- CustomEvents with bubbles: true, composed: true

**Integration**: Update existing services, components, main.ts
- Import from common_pb (not service_pb) for shared types

### 6. Tests

**Unit**:
- Backend: `make test`
- Frontend: `npm run test -- --run`

**E2E**: Create `webserver/web/tests/e2e/[feature].spec.ts`
- Use page.evaluate() for shadow DOM
- Write one, run it, iterate

### 7. Validation (Execute in Order)

```bash
# Build
cd webserver && make build && cd web && npm run build && cd ../..

# Start server
./scripts/dev/start.sh --build &
sleep 20 && curl -sk https://localhost:8443/health

# BDD tests
cd integration/tests/acceptance && go test -v && cd ../../..

# Pre-commit
./scripts/dev/stop.sh
./scripts/githooks/pre-commit.sh

# Pre-push (needs server)
./scripts/dev/start.sh --build &
sleep 20
./scripts/githooks/pre-push
./scripts/dev/stop.sh
```

Fix issues and iterate until all pass.

### 8. Documentation

Follow backlog-and-commit-workflow.md:
1. Read CHANGELOG.md and docs/backlog/infrastructure.md
2. Update CHANGELOG.md - add section at top with [x] items and numbers
3. Update backlog - mark completed [x], update scenario counts
4. Generate commit message (full and short versions)

## Critical Rules

1. Shared proto types go in `common.proto`
2. Use `telemetryService.createSpan` with `span?.` null-safety
3. Follow golang-testing-best-practices.md for Go tests
4. Use file permissions 0600 (not 0644)
5. Import StaticImage from common_pb (not config_service_pb)
6. Object literals for proto in tests: `{...} as Type` (not `new Type()`)
7. Run validation sequence completely
8. Update CHANGELOG and backlog before finishing

## Success Criteria

All must pass:
- Proto generation without errors
- Backend and frontend build
- All unit tests (backend + frontend)
- All BDD scenarios (44+ passing)
- Pre-commit hook
- Pre-push hook
- CHANGELOG and backlog updated
- Commit message generated

## Quick Reference

**Template Variables**:
- `[service]` = file, config, processor
- `[Service]` = File, Config, Processor  
- `[component]` = image-upload, source-drawer
- `[feature]` = upload_image, list_inputs

**Common Patterns**: Reference add-dynamic-feature-end-to-end.md for code examples

**Troubleshooting**:
- Proto fails: Check circular deps, move to common.proto
- BDD fails: Verify server running, check file paths
- Linter fails: Use 0600 permissions, compound operators (>>=)
- Frontend fails: Check common_pb imports, createSpan usage
