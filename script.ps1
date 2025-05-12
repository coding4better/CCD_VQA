# 定义目标文件夹路径
$startPath = "F:\data"  # 替换为你的目标文件夹路径
$outputFile = "F:\data"  # 替换为你想要保存的输出文件路径

if (-not (Test-Path -Path $startPath)) {
    Write-Host "错误：目标路径不存在，请检查路径是否正确。"
    exit
}

# 检查是否有权限访问路径
try {
    Get-ChildItem -Path $startPath -ErrorAction Stop | Out-Null
} catch {
    Write-Host "错误：无法访问目标路径，可能是因为权限不足。请以管理员身份运行 PowerShell。"
    exit
}

# 定义函数，递归获取文件夹结构
function Get-FolderStructure {
    param (
        [string]$path,
        [int]$level = 0
    )
    $indent = " " * $level  # 根据层级生成缩进
    Get-ChildItem -Path $path -Directory | ForEach-Object {
        Write-Output "$indent$($_.Name)" | Out-File -FilePath $outputFile -Append
        Get-FolderStructure -path $_.FullName -level ($level + 1)
    }
}

# 清空输出文件（如果已存在）
if (Test-Path -Path $outputFile) {
    Clear-Content -Path $outputFile
}

# 调用函数，从指定路径开始生成文件夹结构
Get-FolderStructure -path $startPath

Write-Host "文件夹结构已生成并保存到 $outputFile"