rem Import DLL - common::StaticUtility
rem If we are only using StaticUtlity to support ConsoleApplication, then we only import static_utility.dll.

copy R:\Projects\common\bin\static_utility.dll  R:\Projects\common\machine_learning\cpp\neural_network\Debug
copy R:\Projects\common\lib\static_utility.lib  R:\Projects\common\machine_learning\cpp\neural_network\Debug
copy R:\Projects\common\header\static_utility.h R:\Projects\common\machine_learning\cpp\neural_network\test

rem Import DLL - common::ConsoleApplication

copy R:\Projects\common\bin\console_application.dll  R:\Projects\common\machine_learning\cpp\neural_network\Debug
copy R:\Projects\common\lib\console_application.lib  R:\Projects\common\machine_learning\cpp\neural_network\Debug
copy R:\Projects\common\header\console_application.h R:\Projects\common\machine_learning\cpp\neural_network\test