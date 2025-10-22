#!/bin/bash
set -e

echo "Language and Emoji Linter..."

# Files come from environment variable (set by host)
if [ -z "$STAGED_FILES" ]; then
    echo "No source files to check"
    exit 0
fi

CHANGED_FILES="$STAGED_FILES"

ERRORS_FOUND=0

SPANISH_WORDS="función|clase|método|retornar|verdadero|falso|nulo|número|cadena|lista|diccionario|archivo|carpeta|proceso|usuario|contraseña|mensaje|advertencia|éxito|información|configuración|parámetro|resultado|entrada|salida|página|pantalla|botón|campo|formulario|tabla|columna|fila|índice|búsqueda|filtro|ordenar|actualizar|eliminar|guardar|cancelar|aceptar|cerrar|abrir|nuevo|editar|copiar|pegar|cortar|deshacer|rehacer|ejecutar|implementar|inicializar|finalizar|validación|autenticación|autorización|conexión|desconectar|solicitud|respuesta|petición|servidor|cliente|directorio|documento|elemento|contenedor|ventana|plantilla|módulo|componente|servicio|controlador|repositorio|dependencia|biblioteca|paquete|instancia|objeto"

for file in $CHANGED_FILES; do
    [ ! -f "$file" ] && continue
    
    SKIP_EMOJI=0
    [[ "$file" =~ \.(html|css)$ ]] && SKIP_EMOJI=1
    grep -q '(//|#|/\*)\s*emoji-allowed' "$file" 2>/dev/null && SKIP_EMOJI=1
    
    # Skip files that contain nolint:language
    if grep -q "nolint:language" "$file" 2>/dev/null; then
        echo "Skipping $file (contains nolint:language)"
        continue
    fi
    
    # Check Spanish words in file content
    SPANISH_LINES=$(grep -n -i "$SPANISH_WORDS" "$file" || true)
    
    if [ -n "$SPANISH_LINES" ]; then
        echo ""
        echo "ERROR: Spanish words found in $file"
        echo "Only English is allowed in code."
        echo ""
        echo "Problematic lines:"
        echo "$SPANISH_LINES" | head -3
        if [ $(echo "$SPANISH_LINES" | wc -l) -gt 3 ]; then
            echo "... and $(($(echo "$SPANISH_LINES" | wc -l) - 3)) more"
        fi
        ERRORS_FOUND=1
    fi
    
    # Check emojis in file content
    if [ $SKIP_EMOJI -eq 0 ]; then
        EMOJI_LINES=$(perl -C -nle 'print "$.: $_" if /[\x{1F300}-\x{1F9FF}\x{2600}-\x{26FF}\x{2700}-\x{27BF}]/' "$file" || true)
        
        if [ -n "$EMOJI_LINES" ]; then
            echo ""
            echo "ERROR: Emoji found in $file"
            echo "Emojis are only allowed in HTML/CSS files or with '// emoji-allowed' comment."
            echo ""
            echo "Problematic lines:"
            echo "$EMOJI_LINES"
            ERRORS_FOUND=1
        fi
    fi
done

if [ $ERRORS_FOUND -eq 1 ]; then
    echo ""
    echo "FAILED: Language/Emoji validation"
    echo ""
    echo "Tips:"
    echo "  - Use English only in code, comments, and documentation"
    echo "  - Avoid emojis in code (allowed in HTML/CSS)"
    echo "  - To allow emojis in a specific file, add: // emoji-allowed"
    exit 1
fi

echo "Language and Emoji validation passed"

