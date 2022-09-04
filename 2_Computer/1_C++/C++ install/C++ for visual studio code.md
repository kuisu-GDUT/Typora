# C++ for Visual Studio Code

C/C++ support for Visual Studio Code is provided by a Microsoft C/C++ extension to enable cross-platform C and C++ development on Windows, Linux, and macOS.

![cpp extension](.\C++ for visual studio code.assets\cpp-extension.png)

## Install the extension

1. open VS code
2. select the extensions view icon on the activity bar or use the keyboard shortcut(ctrl+shift+x)
3. search for `C++`
4. select Install

After you install the extension, when you open or create `*.cpp` file, you will have syntax highlighting, smart completions and hovers(intelliSense), and error checking

![C++ language features](.\C++ for visual studio code.assets\msg-intellisense.png)

## install a compiler

C++ is a compiled language meaning you program's source code must be translated (compiled) before it can be run on your computer. VS code is first and foremost an editor, and relies on command-line tools to do much of the development workflow. the C++ extension does not include a C++ complier or debugger. You will need to install these tools or use those already installed on your computer.

There may already be a C++ compiler and debugger provided by your academic or work development environment.

## check if you have a compiler installed

Make sure your compiler executable is in your platform path (`%PATH`) on Windows, so that the C++ extension can find it. You can check availability of your C++ tools by opening the Integrated Terminal (`ctrl+`) in VS Code and trying to directly run the compiler.

Checking for the GCC compiler `g++`: `g++ --version`

Checking for the Clang compiler `clang`: `clang --version`

If you don't have a compiler installed, in the example below, we describe how to install the Minimalist GNU for windows (MinGW) C++ tools (compiler and debugger). MinGW is a popular, free toolset for windows.

## Example: install MinGW-X64

We will install Mingw-W64 via [MSYS2](https://www.msys2.org/), which provides up-to-date native builds of GCC, Mingw-w64, and other helpful C++ tools and libraries. [Click here](https://github.com/msys2/msys2-installer/releases/download/2021-06-04/msys2-x86_64-20210604.exe) to download the MSYS2 installer. Then follow the instructions on the [MSYS2 website](https://www.msys2.org/) to install Mingw-w64.

## Add the MinGW compiler to your path

Add Path to your Mingw-w64 `bin` folder to the Windows `PATH` environment variable by using the following steps:

1. In the Windows, search bar, type 'settings' to open your windows setting
2. search for **Edit environment variables for your account**
3. Choose `Path` variable and then select **Edit**
4. Select **New** and add the Mingw-w64 destination folder path, with `\mingw64\bin` appended, to the system path. The exact path depends on which version of Mingw-w64 you have installed and where you installed it. if you used the setting above to install Mingw-w64, then add this to the path `C:msys64\mingw64\bin`
5. Select OK to save the update PATH. You will to reopen any console windows for the new PATH location to be available

## Check MinGW installation

To check that Mingw-w64 tools are correctly installed and available, open a new command prompt and type:

```shell
g++ --version
gdb --version
```

if you don't see the expected output or `g++` or `gdb` is not a recognized command, make sure your PATH entry matches the Mingw-w64 binary location where the compiler tools are located.

## Hello world

To make sure the compiler is installed and configured correctly, we'll create the simplest Hello world C++ program

Create a folder called "hello world" and open vs code in that folder (`code .` opens vs code in the current folder)

```shell
mkdir helloworld
cd helloworld
code .
```

Now create a new file called `helloworld.cpp` with the New File button in the File Explore or File>New File command

![File Explorer New File button](.\C++ for visual studio code.assets\new-file.png)

![helloworld.cpp file](.\C++ for visual studio code.assets\hello-world-cpp.png)

### Add Hello world Source code

Now paste in this source code

```c++
#include <iostream>

int main()
{
    std::cout << "Hello World" << std::endl;
}
```

Now press `ctrl+s` to save the file. You can also enable `Auto save` to automatically save your file changes, by checking Auto Save in the main File menu.

### Build Hello world

Now that we have a simple C++ program, let's build it. Select the `Termial>Run build task` command (ctrl+shift +B) from the main menu.

![Run Build Task menu option](.\C++ for visual studio code.assets\run-build-task.png)

This will display a dropdown with various compiler task options. if you are using a GCC toolset like MinGW, you would choose `C/C++: g++.exe build active file`

![Select g++.exe task](.\C++ for visual studio code.assets\gpp-build-task-msys64.png)

This will compile `helloworld.cpp` and create an executable file called `helloworld.exe`, which will appear in the File Explorer.

![helloworld.exe in the File Explorer](.\C++ for visual studio code.assets\hello-world-exe.png)

### Run hello world

From a command prompt or a new VS Code integrated Terminal, you can now run your program by typing '.\helloworld'.

![Run hello world in the VS Code Integrated Terminal](.\C++ for visual studio code.assets\run-hello-world.png)

If everything is set up correctly, you should see the output "hello world"

## reference

- [C/C++ for visual studio code](https://code.visualstudio.com/docs/languages/cpp#_tutorials)