@echo off
chcp 65001 >nul
echo ============================================================
echo   阶段2: 参数优化 (离线)
echo ============================================================
echo.
echo 选择优化模式:
echo   1. quick    - 快速测试 (约30分钟)
echo   2. standard - 标准优化 (约4-6小时)
echo   3. thorough - 深度优化 (约8-12小时)
echo.
set /p choice="请选择 (1/2/3, 默认2): "

if "%choice%"=="1" (
    set preset=quick
) else if "%choice%"=="3" (
    set preset=thorough
) else (
    set preset=standard
)

echo.
echo 使用预设: %preset%
echo.
pause
python run_optimize.py --preset %preset% --mode full
echo.
echo ============================================================
echo   优化完成!
echo   结果: stock_data/optimization_result.json
echo   报告: stock_data/reports/
echo ============================================================
pause
