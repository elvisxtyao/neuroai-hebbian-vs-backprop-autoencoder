param(
    [int]$DimensionProcessId = 7560
)

$ErrorActionPreference = "Stop"
$python = "D:\Microlearning\.venv\Scripts\python.exe"
$trainingRoot = "C:\Users\86136\AppData\Local\Temp\microlearning-stage3-q5q6-a924932"
$analysisRoot = "D:\Microlearning"
$resultsRoot = "D:\Microlearning\results\formal\phase0_v1_1\stage3_q5q6_sweeps"
$logRoot = "D:\Microlearning\results"
$statusPath = Join-Path $logRoot "stage3_remaining_orchestrator_status.json"

function Write-Status {
    param(
        [string]$Status,
        [string]$Stage,
        [string]$Message = ""
    )
    @{
        status = $Status
        stage = $Stage
        message = $Message
        updated_at = [DateTimeOffset]::UtcNow.ToString("o")
        training_source_commit = "a924932685fd634d0bd054c6171b66859c1c74a2"
        analysis_source_commit = "f30574b217ed404ca073a644bdd22d55cd5a45f9"
    } | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding utf8
}

function Invoke-FormalPython {
    param(
        [string]$WorkingDirectory,
        [string[]]$Arguments,
        [string]$LogName
    )
    $stdout = Join-Path $logRoot "$LogName.stdout.log"
    $stderr = Join-Path $logRoot "$LogName.stderr.log"
    Push-Location $WorkingDirectory
    try {
        & $python -u @Arguments 1> $stdout 2> $stderr
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed ($LASTEXITCODE): $($Arguments -join ' ')"
        }
    }
    finally {
        Pop-Location
    }
}

function Assert-GatePass {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Missing freeze gate: $Path"
    }
    $gate = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    if ($gate.decision -ne "PASS") {
        throw "Freeze gate did not pass: $Path"
    }
}

try {
    Write-Status -Status "running" -Stage "wait_dimension"
    if (Get-Process -Id $DimensionProcessId -ErrorAction SilentlyContinue) {
        Wait-Process -Id $DimensionProcessId
    }
    foreach ($case in @("L16", "L32", "L128")) {
        Assert-GatePass (
            Join-Path $resultsRoot "dimension\$case\freeze_gate.json"
        )
    }

    Write-Status -Status "running" -Stage "architecture_training"
    Invoke-FormalPython `
        -WorkingDirectory $trainingRoot `
        -Arguments @(
            "-m", "training.run_stage3_q5q6_sweeps",
            "--sweep", "architecture"
        ) `
        -LogName "stage3_q6_architecture"
    foreach ($case in @("early_heavy", "late_heavy")) {
        Assert-GatePass (
            Join-Path $resultsRoot "architecture\$case\freeze_gate.json"
        )
    }

    $cases = @(
        @("dimension", "L16"),
        @("dimension", "L32"),
        @("dimension", "L128"),
        @("architecture", "early_heavy"),
        @("architecture", "late_heavy")
    )
    foreach ($entry in $cases) {
        $sweep = $entry[0]
        $case = $entry[1]
        Write-Status `
            -Status "running" `
            -Stage "postfreeze_${sweep}_${case}_test"
        Invoke-FormalPython `
            -WorkingDirectory $trainingRoot `
            -Arguments @(
                "-m", "evaluation.run_stage3_q5q6_test",
                "--sweep", $sweep,
                "--case", $case
            ) `
            -LogName "stage3_${sweep}_${case}_test"

        Write-Status `
            -Status "running" `
            -Stage "postfreeze_${sweep}_${case}_representation"
        Invoke-FormalPython `
            -WorkingDirectory $trainingRoot `
            -Arguments @(
                "-m", "evaluation.run_stage3_q5q6_representation",
                "--sweep", $sweep,
                "--case", $case
            ) `
            -LogName "stage3_${sweep}_${case}_representation"

        Write-Status `
            -Status "running" `
            -Stage "postfreeze_${sweep}_${case}_noise"
        Invoke-FormalPython `
            -WorkingDirectory $trainingRoot `
            -Arguments @(
                "-m", "evaluation.run_stage3_q5q6_noise",
                "--sweep", $sweep,
                "--case", $case
            ) `
            -LogName "stage3_${sweep}_${case}_noise"
    }

    foreach ($case in @("early_heavy", "late_heavy")) {
        Write-Status `
            -Status "running" `
            -Stage "architecture_${case}_updates"
        Invoke-FormalPython `
            -WorkingDirectory $analysisRoot `
            -Arguments @(
                "-m", "evaluation.run_stage3_q4_updates",
                "--config",
                "configs/experiments/stage3_q6_update_${case}_v1.yaml"
            ) `
            -LogName "stage3_architecture_${case}_updates"
    }

    Write-Status -Status "running" -Stage "aggregate_analysis"
    Invoke-FormalPython `
        -WorkingDirectory $analysisRoot `
        -Arguments @(
            "-m", "evaluation.analyze_stage3_q5q6"
        ) `
        -LogName "stage3_q5q6_aggregate"

    Write-Status -Status "running" -Stage "final_tests"
    Invoke-FormalPython `
        -WorkingDirectory $analysisRoot `
        -Arguments @(
            "-m", "pytest", "-q",
            "--junitxml=verification\phase0_v1_1\stage3_final_junit.xml"
        ) `
        -LogName "stage3_final_tests"
    Write-Status -Status "completed" -Stage "all_formal_experiments"
}
catch {
    Write-Status -Status "failed" -Stage "orchestrator" -Message $_.Exception.Message
    throw
}
