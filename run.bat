@echo off
echo Running Core Wars Simulator...
echo.
dotnet run --project src\CoreTrainer.Console\CoreTrainer.Console.csproj -- %*
