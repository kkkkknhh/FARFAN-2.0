import os
import time
from pathlib import Path

# Buscar un plan de ejemplo
plans_dir = "."
for root, dirs, files in os.walk(plans_dir):
    for file in files:
        if file.endswith(('.pdf', '.docx', '.txt')):
            plan_path = os.path.join(root, file)
            file_size = os.path.getsize(plan_path)
            
            print(f"Sample Plan: {file}")
            print(f"Size: {file_size / 1024:.2f} KB ({file_size / (1024*1024):.2f} MB)")
            print(f"Path: {plan_path}")
            
            # Estimar tiempo basado en tamaño
            # Asumiendo ~5MB/min de procesamiento
            estimated_time_min = (file_size / (1024*1024)) / 5
            print(f"Estimated processing time: {estimated_time_min:.2f} minutes")
            print(f"For 170 plans: {estimated_time_min * 170:.2f} minutes ({(estimated_time_min * 170)/60:.2f} hours)")
            break
    break
