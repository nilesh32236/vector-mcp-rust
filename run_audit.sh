#!/bin/bash
# Find all GGUF models automatically
MODELS=$(find ~/.local/share/vector-mcp-rust/models/ -name "*.gguf")

SAMPLES=(
    "src/indexer/chunker.rs"
    "samples/math_utils.go"
    "samples/config_manager.py"
)

mkdir -p benchmarks/ultimate_audit
rm -rf benchmarks/ultimate_audit/*

for model in $MODELS; do
    model_name=$(basename "$model" .gguf)
    echo "========================================================="
    echo "AUDITING: $model_name"
    echo "========================================================="
    
    report_file="benchmarks/ultimate_audit/${model_name}.txt"
    > "$report_file"
    
    for sample in "${SAMPLES[@]}"; do
        echo "   -> Testing $sample..."
        ./target/release/benchmark \
            --model "$model" \
            --test-file "$sample" \
            --output "benchmarks/tmp_report.txt" >> /dev/null 2>&1
        
        if [ $? -eq 0 ]; then
            cat benchmarks/tmp_report.txt >> "$report_file"
            echo -e "\n\n" >> "$report_file"
        else
            echo "   [!] Failed to run $model_name on $sample"
        fi
    done
    
    # Calculate average accuracy if the file isn't empty
    if [ -s "$report_file" ]; then
        accuracy=$(grep "FINAL ACCURACY SCORE" "$report_file" | awk '{print $4}' | sed 's/%//' | awk '{s+=$1} END {if (NR>0) print s/NR; else print 0}')
        echo "   🏆 Overall Multi-Language Accuracy: $accuracy%"
    fi
done

rm benchmarks/tmp_report.txt
echo "========================================================="
echo "ULTIMATE AUDIT COMPLETE. Results in benchmarks/ultimate_audit/"
