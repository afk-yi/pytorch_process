#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SUMMARY_FILE="${SCRIPT_DIR}/benchmark_summary.txt"

# Temp storage for per-version data
declare -A GPU_TIME
declare -A REL_ERR
declare -A VERIFY_RESULT
declare -A NCU_METRICS  # combined key "version|metric_name"
declare -A VERSIONS

version_order=()

# ---- Phase 1: Compile and profile each version ----
for dir in "$SCRIPT_DIR"/reduce_v*/; do
    [ -d "$dir" ] || continue
    dir_name="$(basename "$dir")"
    cu_file=$(find "$dir" -maxdepth 1 -name '*.cu' | head -1)
    [ -z "$cu_file" ] && continue

    binary_path="${dir}/${dir_name}"

    echo "===== $dir_name ====="
    echo "  Compiling..."
    nvcc -O2 -lineinfo -o "$binary_path" "$cu_file" 2>&1
    [ ! -f "$binary_path" ] && { echo "  [SKIP] Compilation failed"; continue; }

    echo "  Profiling (ncu)..."
    ncu --section-folder /usr/local/cuda-12.4/nsight-compute-2024.1.0/sections \
        --metrics \
dram__throughput.avg,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum.per_second,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes_read.sum,\
dram__bytes_write.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l2tex__bytes_read.sum,\
sm__warps_active.avg.pct_of_peak_sustained_elapsed,\
smsp__pcsamp_warps_issue_stalled_barrier,\
smsp__pcsamp_warps_issue_stalled_memory_dependency \
        --target-processes all \
        "$binary_path" \
        > "${dir}/ncu_stdout.txt" 2>&1

    echo "  Profiling (nvprof)..."
    nvprof "$binary_path" > "${dir}/nvprof_stdout.txt" 2>&1

    # Extract data from ncu stdout
    gpu_time=$(grep -oP 'GPU 耗时:\s*\K[\d.]+' "${dir}/ncu_stdout.txt" | head -1)
    rel_err=$(grep -oP 'rel_err=\K[\d.e\-]+' "${dir}/ncu_stdout.txt" | head -1)
    verify_result=$(grep -oP '结果验证:\s*\K[✅❌]+' "${dir}/ncu_stdout.txt" | head -1)

    version_order+=("$dir_name")
    VERSIONS["$dir_name"]=1
    GPU_TIME["$dir_name"]=$gpu_time
    REL_ERR["$dir_name"]=$rel_err
    VERIFY_RESULT["$dir_name"]=$verify_result

    # Extract NCU metrics from ncu_stdout.txt
    # Metric lines look like: "    metric_name ... unit   value"
    ncu_out="${dir}/ncu_stdout.txt"
    for metric in dram__throughput.avg \
                  l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum.per_second \
                  sm__throughput.avg.pct_of_peak_sustained_elapsed \
                  dram__bytes_read.sum \
                  dram__bytes_write.sum \
                  l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum \
                  l2tex__bytes_read.sum \
                  sm__warps_active.avg.pct_of_peak_sustained_elapsed \
                  smsp__pcsamp_warps_issue_stalled_barrier \
                  smsp__pcsamp_warps_issue_stalled_memory_dependency; do
        # Extract the value (last whitespace-separated field) from lines matching the metric
        val=$(awk "/^[[:space:]]+${metric}[[:space:]]+/ { v = \$NF } END { print v }" "$ncu_out")
        NCU_METRICS["${dir_name}|${metric}"]=$val
    done

    echo ""
done

# ---- Phase 2: Generate summary file ----
rm -f "$SUMMARY_FILE"

# Header
printf "%-58s" "指标" >> "$SUMMARY_FILE"
for ver in "${version_order[@]}"; do
    printf "  %-14s" "$ver" >> "$SUMMARY_FILE"
done
printf "\n" >> "$SUMMARY_FILE"

# Separator
printf "%58s" "" | tr ' ' '=' >> "$SUMMARY_FILE"
printf "  " >> "$SUMMARY_FILE"
for ver in "${version_order[@]}"; do
    printf "%16s" "" | tr ' ' '=' >> "$SUMMARY_FILE"
done
printf "\n" >> "$SUMMARY_FILE"

echo "=== NCU ===" >> "$SUMMARY_FILE"

# NCU metric rows
write_row() {
    local label="$1" suffix="$2"
    printf "%-58s" "$label" >> "$SUMMARY_FILE"
    for ver in "${version_order[@]}"; do
        local key="${3:-${ver}}"
        local val="${NCU_METRICS[${ver}|${4}]}"
        printf "  %14s" "$val" >> "$SUMMARY_FILE"
    done
    printf "\n" >> "$SUMMARY_FILE"
}

write_str_row() {
    local label="$1" suffix="$2"
    printf "%-58s" "$label" >> "$SUMMARY_FILE"
    for ver in "${version_order[@]}"; do
        printf "  %14s" "${3}" >> "$SUMMARY_FILE"
    done
    printf "\n" >> "$SUMMARY_FILE"
}

# Row: GPU 耗时
printf "%-58s" "GPU 耗时 (ms)" >> "$SUMMARY_FILE"
for ver in "${version_order[@]}"; do
    printf "  %14s" "${GPU_TIME[$ver]}" >> "$SUMMARY_FILE"
done
printf "\n" >> "$SUMMARY_FILE"

# Row: 结果验证
printf "%-58s" "结果验证" >> "$SUMMARY_FILE"
for ver in "${version_order[@]}"; do
    printf "  %14s" "${VERIFY_RESULT[$ver]}" >> "$SUMMARY_FILE"
done
printf "\n" >> "$SUMMARY_FILE"

# Row: 相对误差
printf "%-58s" "相对误差" >> "$SUMMARY_FILE"
for ver in "${version_order[@]}"; do
    printf "  %14s" "${REL_ERR[$ver]}" >> "$SUMMARY_FILE"
done
printf "\n" >> "$SUMMARY_FILE"

# NCU metrics
nc_metrics=(
    "dram__bytes_read.sum"
    "dram__bytes_write.sum"
    "dram__throughput.avg"
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum.per_second"
    "l2tex__bytes_read.sum"
    "sm__throughput.avg.pct_of_peak_sustained_elapsed"
    "sm__warps_active.avg.pct_of_peak_sustained_elapsed"
    "smsp__pcsamp_warps_issue_stalled_barrier"
    "smsp__pcsamp_warps_issue_stalled_memory_dependency"
)

for metric in "${nc_metrics[@]}"; do
    printf "%-58s" "$metric" >> "$SUMMARY_FILE"
    for ver in "${version_order[@]}"; do
        val="${NCU_METRICS[${ver}|${metric}]}"
        [ -z "$val" ] && val="n/a"
        printf "  %14s" "$val" >> "$SUMMARY_FILE"
    done
    printf "\n" >> "$SUMMARY_FILE"
done

printf "\n" >> "$SUMMARY_FILE"
echo "=== nvprof 原始输出 ===" >> "$SUMMARY_FILE"
printf "\n" >> "$SUMMARY_FILE"

for ver in "${version_order[@]}"; do
    echo "===== $ver =====" >> "$SUMMARY_FILE"
    cat "${SCRIPT_DIR}/${ver}/nvprof_stdout.txt" >> "$SUMMARY_FILE"
    printf "\n" >> "$SUMMARY_FILE"
done

echo "All benchmarks done. Summary written to $SUMMARY_FILE"
