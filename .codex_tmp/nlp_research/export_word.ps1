$ErrorActionPreference = 'Stop'
$reportPath = 'C:\Transformer-rl-retro-propulsion\Plan\NLP\aviation_nlp_embedding_project_ideas.docx'
$reviewDir = 'C:\Transformer-rl-retro-propulsion\.codex_tmp\nlp_research\render'
[void](New-Item -ItemType Directory -Force -Path $reviewDir)
$reviewPdf = Join-Path $reviewDir 'aviation_nlp_embedding_project_ideas.pdf'
$wordApp = $null
$reportDoc = $null
try {
    $wordApp = New-Object -ComObject Word.Application
    $wordApp.Visible = $false
    $wordApp.DisplayAlerts = 0
    $reportDoc = $wordApp.Documents.Open($reportPath, $false, $true)
    $reportDoc.Repaginate()
    $reportDoc.ExportAsFixedFormat($reviewPdf, 17)
    Write-Output "Rendered PDF: $reviewPdf"
    Write-Output "Pages: $($reportDoc.ComputeStatistics(2))"
    Write-Output "Native equations: $($reportDoc.OMaths.Count)"
} finally {
    if ($null -ne $reportDoc) { $reportDoc.Close(0); [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($reportDoc) }
    if ($null -ne $wordApp) { $wordApp.Quit(); [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($wordApp) }
}
