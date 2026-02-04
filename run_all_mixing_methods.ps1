# Run All Mixing Methods - Sequential Comparison
# PowerShell script to run all four mixing methods one after another

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "RUNNING ALL MIXING METHODS SEQUENTIALLY" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

$config = @{
    NumClients = 10
    Rounds = 15
    Epochs = 5
    Dataset = "cifar10"
}

Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Clients: $($config.NumClients)"
Write-Host "  Rounds: $($config.Rounds)"
Write-Host "  Epochs: $($config.Epochs)"
Write-Host "  Dataset: $($config.Dataset)"
Write-Host ""

$methods = @(
    @{Name="metropolis_hastings"; Desc="Metropolis-Hastings (Default)"},
    @{Name="max_degree"; Desc="Max-Degree (Uniform Weights)"},
    @{Name="jaccard"; Desc="Jaccard Similarity"},
    @{Name="matcha"; Desc="MATCHA (Optimal)"}
)

$results = @()
$totalStart = Get-Date

foreach ($i in 0..($methods.Count-1)) {
    $method = $methods[$i]
    $num = $i + 1
    
    Write-Host ""
    Write-Host "======================================================================" -ForegroundColor Green
    Write-Host "[$num/4] Running: $($method.Desc)" -ForegroundColor Green
    Write-Host "======================================================================" -ForegroundColor Green
    Write-Host ""
    
    $expName = "$($method.Name)_comparison"
    $methodStart = Get-Date
    
    $cmd = "python main.py --type decentralized --mixing_method $($method.Name) " +
           "--num_clients $($config.NumClients) --rounds $($config.Rounds) " +
           "--epochs $($config.Epochs) --dataset $($config.Dataset) " +
           "--experiment_name $expName"
    
    try {
        Invoke-Expression $cmd
        $status = "SUCCESS"
        Write-Host "`n$($method.Desc) completed successfully!" -ForegroundColor Green
    }
    catch {
        $status = "FAILED"
        Write-Host "`n$($method.Desc) failed!" -ForegroundColor Red
    }
    
    $methodEnd = Get-Date
    $duration = ($methodEnd - $methodStart).TotalMinutes
    
    $results += [PSCustomObject]@{
        Method = $method.Name
        Description = $method.Desc
        Status = $status
        ExperimentName = $expName
        Duration = $duration
    }
    
    Write-Host "Duration: $([math]::Round($duration, 1)) minutes" -ForegroundColor Cyan
}

$totalEnd = Get-Date
$totalDuration = ($totalEnd - $totalStart).TotalMinutes

# Summary
Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "EXECUTION SUMMARY" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

foreach ($result in $results) {
    $statusColor = if ($result.Status -eq "SUCCESS") { "Green" } else { "Red" }
    $statusSymbol = if ($result.Status -eq "SUCCESS") { "[✓]" } else { "[✗]" }
    Write-Host "$statusSymbol " -NoNewline -ForegroundColor $statusColor
    Write-Host "$($result.Description.PadRight(35)) " -NoNewline
    Write-Host "$($result.Status.PadRight(10)) " -NoNewline
    Write-Host "logs/$($result.ExperimentName)/" -ForegroundColor Gray
}

Write-Host ""
Write-Host "Total time: $([math]::Round($totalDuration, 1)) minutes" -ForegroundColor Yellow

# Next steps
Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "NEXT STEPS - ANALYZE RESULTS" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Compare results in logs/ directory:" -ForegroundColor Yellow
foreach ($result in $results) {
    Write-Host "   - logs/$($result.ExperimentName)/" -ForegroundColor Gray
}

Write-Host ""
Write-Host "2. Visualize with analysis notebook:" -ForegroundColor Yellow
Write-Host "   jupyter notebook analysis.ipynb" -ForegroundColor Gray

Write-Host ""
Write-Host "3. Quick comparison command:" -ForegroundColor Yellow
Write-Host '   python -c "import pandas as pd; [print(f''{m}: {pd.read_csv(f''logs/{m}_comparison/*/round_summary.csv'').avg_accuracy.iloc[-1]:.2f}%'') for m in [''metropolis_hastings'', ''max_degree'', ''jaccard'', ''matcha'']]"' -ForegroundColor Gray

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
