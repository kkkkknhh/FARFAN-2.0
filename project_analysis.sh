#!/bin/bash
echo "=== PROYECTO COMPLETO ===" > PROJECT_FULL_INFO.txt
echo "" >> PROJECT_FULL_INFO.txt

echo "1. ESTRUCTURA DE ARCHIVOS:" >> PROJECT_FULL_INFO.txt
find . -type f -name "*.py" -o -name "*.txt" -o -name "*.md" -o -name "*.json" -o -name "*.yaml" -o -name "*.yml" -o -name "requirements*" | head -100 >> PROJECT_FULL_INFO.txt

echo -e "\n2. ARCHIVOS PRINCIPALES (primeras 50 líneas):" >> PROJECT_FULL_INFO.txt
for file in orchestrator.py main.py app.py server.py run.py *.py; do
  if [ -f "$file" ]; then
    echo -e "\n--- $file ---" >> PROJECT_FULL_INFO.txt
    head -50 "$file" >> PROJECT_FULL_INFO.txt
  fi
done

echo -e "\n3. README/DOCS:" >> PROJECT_FULL_INFO.txt
cat README.md 2>/dev/null >> PROJECT_FULL_INFO.txt || echo "No README" >> PROJECT_FULL_INFO.txt
cat IMPLEMENTATION_SUMMARY.txt 2>/dev/null >> PROJECT_FULL_INFO.txt || echo "No IMPLEMENTATION_SUMMARY" >> PROJECT_FULL_INFO.txt

echo -e "\n4. REQUIREMENTS:" >> PROJECT_FULL_INFO.txt
cat requirements.txt 2>/dev/null >> PROJECT_FULL_INFO.txt || echo "No requirements.txt" >> PROJECT_FULL_INFO.txt

echo -e "\n5. CONFIGURACIÓN:" >> PROJECT_FULL_INFO.txt
cat config*.yaml config*.json 2>/dev/null >> PROJECT_FULL_INFO.txt || echo "No config files" >> PROJECT_FULL_INFO.txt

echo -e "\n6. BUCKET S3:" >> PROJECT_FULL_INFO.txt
aws s3 ls s3://sin-carreta-plans-2025/plans/ --recursive | head -20 >> PROJECT_FULL_INFO.txt

echo -e "\n7. IMPORTS Y DEPENDENCIAS:" >> PROJECT_FULL_INFO.txt
grep -h "^import \|^from " *.py 2>/dev/null | sort -u >> PROJECT_FULL_INFO.txt

cat PROJECT_FULL_INFO.txt
