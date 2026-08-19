$c = Get-Content "C:\Users\MASTER CORE TI\project\poultry_paper\training_output.txt"
# Find model headers
for ($i = 0; $i -lt $c.Count; $i++) {
    if ($c[$i] -match "^\[MODEL\]" -or $c[$i] -match "^Training [a-z]" -or $c[$i] -match "^\[3/6\]" -or $c[$i] -match "^\[4/6\]" -or $c[$i] -match "^\[5/6\]" -or $c[$i] -match "^\[6/6\]") {
        Write-Host "${i}: $($c[$i])"
    }
}
