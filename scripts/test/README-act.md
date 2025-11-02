# Cómo usar act para probar workflows localmente

## Requisitos

1. **Token de GitHub Personal Access Token (PAT)**
   
   Necesitas crear un token en: https://github.com/settings/tokens/new
   
   Permisos necesarios:
   - `public_repo` (si tu repo es público)
   - `repo` completo (si tu repo es privado)
   
   Una vez creado, agrégalo a `.secrets.act`:
   ```
   GITHUB_TOKEN=ghp_tu_token_aqui
   ```

2. **Docker corriendo**
   ```bash
   docker ps
   ```

## Uso

### Listar workflows disponibles
```bash
./scripts/test/workflow-local.sh --list
```

### Ejecutar workflow completo
```bash
./scripts/test/workflow-local.sh
```

### Ejecutar job específico
```bash
./scripts/test/workflow-local.sh --job build
```

## Configuración

El archivo `.actrc` ya está configurado con:
- Cache de acciones (`--use-new-action-cache`)
- Platform correcta para Ubuntu
- Secret file apuntando a `.secrets.act`

## Notas

- La primera vez que ejecutes act, descargará las acciones y las cacheará
- El cache está en `~/.cache/act`
- Si tienes problemas de autenticación, verifica que tu token sea válido
