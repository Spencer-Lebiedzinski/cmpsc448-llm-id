# Repeat the RQ1/RQ2 runs and the RQ4 ablations with another seed (seed 42 lives in runs/).
# Usage: scripts\run_seeds.ps1 -Seed 43 [-Py path\to\python.exe]
param([int]$Seed = 43, [string]$Py = "python")
$env:PYTHONUTF8 = "1"
$R = "runs_seeds/seed$Seed"
foreach ($m in 'cnn', 'rnn') {
    foreach ($mode in 'output', 'input', 'both') {
        & $Py -W ignore -m src.train --model $m --input_mode $mode --seed $Seed --runs_dir $R
    }
    foreach ($a in 'mask_identity', 'strip_format', 'truncate64', 'mask_identity,strip_format') {
        & $Py -W ignore -m src.train --model $m --input_mode output --ablation $a --seed $Seed --runs_dir $R
    }
}
