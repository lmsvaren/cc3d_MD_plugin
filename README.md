# cc3d_MD_plugin
MacOS cc3d compilation instructions [Here](https://compucell3ddevelopersmanual.readthedocs.io/en/master/building_core_cc3d_cpp_code_mac.html)

[YouTube: Michael Getz (2026) C++ Extensions](https://www.youtube.com/watch?v=6xor9AhwINY)

Files after creating new PlugIn module

Twedit > CC3DC++ > Generate New Module > Module Core Name: AdhesiveSat

|File|Notes|
|---|---|
|AdhesiveSatPluginProxy.cpp|Define how plugin is defined in the XML file, plugin description|
|AdhesiveSatPlugin.h|Declaring everything|
|AdhesiveSatPlugin.cpp|Defining everything|

|CMakesList|NA|

# Helpful VIM
VIM is a common text editor used to edit files within command line. [Here](https://vim-adventures.com/) is a fun game to practice VIM commands.
|Command|Function|
|---|---|
|`vi file.txt`|open up file.txt in the editor|
|`i`|'insert' to be able to start editing the file|
|`dd`|delete the current line|
|`esc`+`:q`|exit editor after making **NO** changes|
|`esc`+`:q!`|exit editor and discard any changes made|
|`esc`+`:wq`|exit editor and save file after making changes|
