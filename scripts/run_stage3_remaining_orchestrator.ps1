param(
    [int]$DimensionProcessId = 0,
    [string]$TrainingRoot = $env:MICROLEARNING_TRAINING_ROOT
)

$ErrorActionPreference = "Stop"
$analysisRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($TrainingRoot)) {
    throw "Pass -TrainingRoot or set MICROLEARNING_TRAINING_ROOT to the historical training worktree."
}
$python = Join-Path $analysisRoot ".venv\Scripts\python.exe"
$trainingRoot = $TrainingRoot
$resultsRoot = Join-Path $analysisRoot "results\formal\phase0_v1_1\stage3_q5q6_sweeps"
$logRoot = Join-Path $analysisRoot "results"
$statusPath = Join-Path $logRoot "stage3_remaining_orchestrator_status.json"
$analysisSourceCommit = (
    & git `
        -c "safe.directory=$($analysisRoot.Replace('\', '/'))" `
        -C $analysisRoot `
        rev-parse HEAD
).Trim()

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
        analysis_source_commit = $analysisSourceCommit
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
    $previousErrorAction = $ErrorActionPreference
    try {
        # Windows PowerShell 5.1 promotes native stderr to an ErrorRecord when
        # ErrorActionPreference is Stop. Scientific libraries legitimately
        # emit warnings on stderr, so only the native exit code is a failure.
        $ErrorActionPreference = "Continue"
        & $python -u @Arguments 1> $stdout 2> $stderr
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorAction
        Pop-Location
    }
    if ($exitCode -ne 0) {
        throw (
            "Command failed ($exitCode): " +
            ($Arguments -join " ")
        )
    }
}

function Test-JsonField {
    param(
        [string]$Path,
        [string]$Field,
        $Expected
    )
    if (-not (Test-Path -LiteralPath $Path)) {
        return $false
    }
    $record = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    return $record.$Field -eq $Expected
}

function Move-IncompleteOutput {
    param(
        [string]$Path,
        [string]$Stage
    )
    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }
    $resolvedRoot = [IO.Path]::GetFullPath($resultsRoot)
    $resolvedPath = [IO.Path]::GetFullPath($Path)
    if (-not $resolvedPath.StartsWith(
        $resolvedRoot + [IO.Path]::DirectorySeparatorChar,
        [StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Refusing to quarantine output outside results root: $Path"
    }
    $recoveryRoot = Join-Path $resultsRoot "_recovery"
    New-Item -ItemType Directory -Path $recoveryRoot -Force | Out-Null
    $timestamp = [DateTimeOffset]::UtcNow.ToString("yyyyMMddTHHmmssfffZ")
    $destination = Join-Path $recoveryRoot "${Stage}_${timestamp}"
    Move-Item -LiteralPath $resolvedPath -Destination $destination
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
    if (
        $DimensionProcessId -gt 0 -and
        (Get-Process -Id $DimensionProcessId -ErrorAction SilentlyContinue)
    ) {
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
        $caseRoot = Join-Path $resultsRoot "$sweep\$case"
        $testRoot = Join-Path $caseRoot "test_evaluation"
        $testComplete = Test-JsonField `
            -Path (Join-Path $testRoot "summary.json") `
            -Field "records_complete" `
            -Expected $true
        if (-not $testComplete) {
            Move-IncompleteOutput `
                -Path $testRoot `
                -Stage "${sweep}_${case}_test"
            Invoke-FormalPython `
                -WorkingDirectory $trainingRoot `
                -Arguments @(
                    "-m", "evaluation.run_stage3_q5q6_test",
                    "--sweep", $sweep,
                    "--case", $case
                ) `
                -LogName "stage3_${sweep}_${case}_test"
        }

        Write-Status `
            -Status "running" `
            -Stage "postfreeze_${sweep}_${case}_representation"
        $representationRoot = Join-Path $caseRoot "representation"
        $representationComplete = Test-JsonField `
            -Path (Join-Path $representationRoot "integrity.json") `
            -Field "record_count" `
            -Expected 20
        if (-not $representationComplete) {
            Move-IncompleteOutput `
                -Path $representationRoot `
                -Stage "${sweep}_${case}_representation"
            Invoke-FormalPython `
                -WorkingDirectory $trainingRoot `
                -Arguments @(
                    "-m", "evaluation.run_stage3_q5q6_representation",
                    "--sweep", $sweep,
                    "--case", $case
                ) `
                -LogName "stage3_${sweep}_${case}_representation"
        }

        Write-Status `
            -Status "running" `
            -Stage "postfreeze_${sweep}_${case}_noise"
        $noiseRoot = Join-Path $caseRoot "noise"
        $noiseComplete = Test-JsonField `
            -Path (Join-Path $noiseRoot "integrity.json") `
            -Field "checkpoint_count" `
            -Expected 20
        if (-not $noiseComplete) {
            Move-IncompleteOutput `
                -Path $noiseRoot `
                -Stage "${sweep}_${case}_noise"
            Invoke-FormalPython `
                -WorkingDirectory $trainingRoot `
                -Arguments @(
                    "-m", "evaluation.run_stage3_q5q6_noise",
                    "--sweep", $sweep,
                    "--case", $case
                ) `
                -LogName "stage3_${sweep}_${case}_noise"
        }
    }

    foreach ($case in @("early_heavy", "late_heavy")) {
        Write-Status `
            -Status "running" `
            -Stage "architecture_${case}_updates"
        $updateRoot = Join-Path `
            $resultsRoot `
            "architecture\$case\update_mechanisms"
        $updateComplete = Test-JsonField `
            -Path (Join-Path $updateRoot "integrity.json") `
            -Field "formal_update_rows" `
            -Expected 90
        if (-not $updateComplete) {
            Move-IncompleteOutput `
                -Path $updateRoot `
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
    }

    Write-Status -Status "running" -Stage "aggregate_analysis"
    $analysisRootPath = Join-Path $resultsRoot "analysis"
    $analysisComplete = Test-JsonField `
        -Path (Join-Path $analysisRootPath "integrity.json") `
        -Field "all_values_finite" `
        -Expected $true
    if (-not $analysisComplete) {
        Move-IncompleteOutput `
            -Path $analysisRootPath `
            -Stage "aggregate_analysis"
        Invoke-FormalPython `
            -WorkingDirectory $analysisRoot `
            -Arguments @(
                "-m", "evaluation.analyze_stage3_q5q6"
            ) `
            -LogName "stage3_q5q6_aggregate"
    }

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
