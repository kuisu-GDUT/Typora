# Using GCC with MinGW

In this tutorial, you configure VS Code to use the GCC C++ compiler and GDB debugger from [mingw-w64](http://mingw-w64.org/) to create programs that run on windows.

After configuring VS Code, you will compile and debug a simple hello world program in VS Code. 

## Prerequisties

To successfully complete this tutorial, we must do the following steps:

1. install VS Code
2. install the [C/C++ extension for VS Code](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools).

![C/C++ extension](.\using GCC with MinGW.assets\cpp-extension.png)

3. Get the latest version of Ming-GW via [MSYS2](https://www.msys2.org/), which provides up-to-date native builds of GCC, Mingw-w64, and other helpful C++ tools and libraries. 
4. Add the path to your Mingw-w64 `bin` folder to the windows `PATH` environment variable by using the following steps:

## Check MinGW installation

To check that Mingw-w64 tools are correctly installed and available, open a new command prompt and type:

```shell
g++ --version
gdb --version
```

If you don't see the excepted output or `g++` or `gdb` is not a recognized command, make sure `PATH` entry matches th Mingw-w64 binary location where the compilers are located.

### Create Hello world

From a Windows command prompt, create an empty folder called `projects` where you can place all VS Code projects. Then create a sub-folder called `helloworld`, navigate into it, and open VS Code in that folder by entering the following commands.

```shell
mkdir projects
cd projects
mkdir helloworld
cd helloworld
code .
```

The 'code .' command opens VS Code in the current working folder, which becomes 'workspace'. As you go through the tutorial, you will see three files created in a `.vscode` folder in the workspace:

- `tasks.json`: build instructions
- `launch.json`: debugger settings
- `c_cpp_properties,json`: compiler path and IntelliSense settings

### add a source code file

In the file Explorer title bar, select the new file button and name the file `helloworld.cpp`

![New File title bar button](.\using GCC with MinGW.assets\new-file-button.png)

### Add hello world source code

Now paste in this source code:

```c++
#include <iostream>
#include <vector>
#include <string>

using namespace std;

int main()
{
    vector<string> msg {"Hello", "C++", "World", "from", "VS Code", "and the C++ extension!"};

    for (const string& word : msg)
    {
        cout << word << " ";
    }
    cout << endl;
}
```

Now press `ctrl+s` to save the file. Notice how the file you just added appears in the File Exporer view (ctrl+shift+E) in the side bar of VS Code.

![File Explorer](.\using GCC with MinGW.assets\file-explorer-mingw.png)

### Explore InterlliSense

In our new `helloworld.cpp` file, hover over `vector` or `string` to see type information. After the declaration of the `msg` variable, start typing `msg`. as you would when calling a member function. we should immediately see a completion list that shows all the member function, and a window that shows the type information for the msg object:

![Statement completion IntelliSense](.\using GCC with MinGW.assets\msg-intellisense.png)

we can press the `Tab` key to insert the selected member; then, when you add the opening parenthesis, you will see information about any arguments that the function requires.

## Build helloworld.cpp

Next, we will create a `tasks.json` file to tell VS Code how to build (compile) the program. This task will invoke the g++ compiler to create an executable file based on the source code.

From the main menu, choose `Terminal > Configure Default Build Task`. In the dropdown, which will display a stask dropdown listing various predefined build tasks for C++ compilers. Choose **g++.exe build active file**, which will build the file that is currently displayed (active) in the editor.

![Tasks C++ build dropdown](.\using GCC with MinGW.assets\build-active-file.png)

This will create `tasks.json` file in a `.vscode` folder and open it in the editor. our new `tasks.json` file should look similar to the JSON below:

```json
{
  "tasks": [
    {
      "type": "cppbuild",
      "label": "C/C++: g++.exe build active file",
      "command": "C:/msys64/mingw64/bin/g++.exe",
      "args": ["-g", "${file}", "-o", "${fileDirname}\\${fileBasenameNoExtension}.exe"],
      "options": {
        "cwd": "${fileDirname}"
      },
      "problemMatcher": ["$gcc"],
      "group": {
        "kind": "build",
        "isDefault": true
      },
      "detail": "compiler: C:/msys64/mingw64/bin/g++.exe"
    }
  ],
  "version": "2.0.0"
}
```

The `command` setting specifies the program to run; in this case that is g++. the `args` array specifies the command-line arguments that will be passed to g++. These arguments must be specified in the order expected by the compiler. This task tell g++ to take active file (`${file}`), compile it, and create an executable file in the current directory (`${fileDirname}`) with the same name as the active file but with the `.exe` for our example.

The `label` value is what we will see in the task list; we can name this whatever you like. 

The `"isDefault": true` value in the `group` object specifies that this task will be run when you press `ctrl+shift+B`. This property is for convenience only; if you set it to false, you can still run it from the Terminal menu with **Tasks: Run Build Task**

### Run the build

1. Go back to `helloworld.cpp`. our task build the active file and we want to build `helloworld`.
2. To run the build task defined in `tasks.json`, press `ctrl+shift+B` or from the Terminal main menu choose Run Build Task.
3. When the task starts, we should see the integrated Terminal panel appear below the source code editor. After the task completes, the terminal shows output from the compiler that indicates whether the build succeeded or failed. For a successful g++ build, the output looks something like this:

![G++ build output in terminal](.\using GCC with MinGW.assets\mingw-build-output.png)

4. Create a new terminal using the + button and you will have a new terminal with the `hellowrold`  folder as the working directory. Run `dir` and we should now see the executable `helloworld.exe`

![Hello World in PowerShell terminal](.\using GCC with MinGW.assets\mingw-dir-output.png)

5. we can run `helloworld` in the terminal by typing `helloworld.exe` 

### Modifying tasks.json

we can modify our `tasks.json` to build multiple C++ file by using an argument like `"${workspaceFloder}\\*.cpp"` instead of `${file}.` This will build all `.cpp` file in our current folder. We can also modify the output filename by replacing

`"${fileDirname}\\${fileBasenameNoExtension}.exe"` with hard-coded filename (for example `"${workspaceFolder}\\myProgram.exe"`)



## Debug hellowrld.cpp

next, we will create a `launch.json` file to configures VS Code to launch the GDB debugger when we press `F5` to debug the program.

1. From the main menu, choose `Run > Add configuration ...` and choose **C++(GDB/LLDB**.
2. we'll then see a dropdown for various predefined debugging configurations, choose **g++.exe build and debug active file**

![C++ debug configuration dropdown](.\using GCC with MinGW.assets\build-and-debug-active-file.png)

VS Code create a `launch.json` file, opens it in the editor, and builds and runs 'helloworld'

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "g++.exe - Build and debug active file",
      "type": "cppdbg",
      "request": "launch",
      "program": "${fileDirname}\\${fileBasenameNoExtension}.exe",
      "args": [],
      "stopAtEntry": false,
      "cwd": "${fileDirname}",
      "environment": [],
      "externalConsole": false,
      "MIMode": "gdb",
      "miDebuggerPath": "C:\\msys64\\mingw64\\bin\\gdb.exe",
      "setupCommands": [
        {
          "description": "Enable pretty-printing for gdb",
          "text": "-enable-pretty-printing",
          "ignoreFailures": true
        }
      ],
      "preLaunchTask": "C/C++: g++.exe build active file"
    }
  ]
}
```

The `program` setting specifies the program we want it debug. Here it is set to the active file folder `${fileDirname}$` and active filename with the `.exe` extension `${fileBasenameNoExtension}.exe`, which if `helloworld.cpp` is active file will be `helloworld.exe`

By default, the C++ extension won't add any breakpoints to your source code and the `stopAtEntry` value is set to `false`.

Change the `stopAtEntry` value to `true` to cause the debugger to stop on the `main` method when we start debugging

> Note: The `preLaunchTask` setting is used to specify task to executed before launch. Make sure it is consistent with `tasks.json` file `label` setting.

### Start a debugging session

1. Go back to `helloworld.cpp` so that it is the active file.
2. Pres `F5` or from the main menu choose `Run > Start Debugging`. Before we start stepping through the source code, let's take a moment to notice several changes in the user interface:

- The integrated Terminal appears at the bottom of the source code editor. In the **Debug Output** tab, we see output that indicates the debugger is up and running.
- The editor highlights the first statement in the `main` method. This is a breakpoint that the C++ extension automatically sets for you:

![Initial breakpoint](.\using GCC with MinGW.assets\stopAtEntry.png)

- The Run view on the left shows debugging information. We'll see an example later in the tutorial
- At the top of the code editor, a debugging control panel appears. we can move this around the screen by grabbing the dots on the left side.

## Step through the code

Now we're ready to start stepping through the code.

1. Click or press the Step over icon in the debugging control panel.

![Step over button](.\using GCC with MinGW.assets\step-over-button.png)

This will advance program execution to the first line of the for loop, and skip over all the internal function calls within the `vector` and `string` classes that are invoked when the `msg` variable is created and initialized. Notice the change in the Variables window on the left.

![Debugging windows](.\using GCC with MinGW.assets\debug-view-variables.png)

In this case, the errors are expected because, although the variable names for the loop are now visible to the debugger, the statement has not executed yet, so there is noting to read at this point. The content of `msg` are visible, however, because that statement has completed.

2. Press **Step over** again to advance to the next statement in this program (skipping over all the internal code that is executed to initialize the loop). Now, the **Variables** window shows information about the loop variables.
3. Press **Step over** again to execute the `cout` statement. (Not that as of the March 2019 release, the C++ extension does not print any output to the Debug console until the loop exits)
4. If we like, we can keep pressing **Step over** until all the words in the vector have been printed to the console. But if we are curious, try pressing the **Step Into** button to step through source code in the C++ standard library!

![Breakpoint in gcc standard library header](.\using GCC with MinGW.assets\gcc-system-header-stepping.png)

 To return to our own code, one way is to keep pressing Step over. Another way is to set a breakpoint in our code by switching to the `helloworld.cpp` tab in the code editor, putting the insertion point somewhere on the `cout` statement inside the loop, and pressing `F9`. A red dot appears in the gutter on the left to indicate that a breakpoint has been set on this line

![Breakpoint in main](.\using GCC with MinGW.assets\breakpoint-in-main.png)

Then press `F5` to start execution from the current line in the standard library header. Execution will break on `cout`. If we want, we can press `F9` again to toggle off the breakpoint

When the loop has completed, we can see the output in the integrated Terminal, along with some other diagnostic information that is output by GDB.

![Debug output in terminal](.\using GCC with MinGW.assets\mingw-debug-output.png)

## set a watch

Sometimes we might want to keep track of the value of a variables as our program executes. We can do this by setting a **watch** on the variable.

1. Place the insertion point inside the loop. In the **watch** window, click the plus sign and in the text box, type `word`, which is the name of the loop variable. Now view the Watch window as we step through the loop.

![Watch window](.\using GCC with MinGW.assets\watch-window.png)

2. Add another watch by adding this statement  before the loop: `int i =0`. Then, inside the loop, add this statement: `++i`. Now add a watch for `i` as we did in the previous step.

3. To quickly view the value of any variable while execution is paused on a breakpoint, we can hover over it  with the mouse pointer.

   ![Mouse hover](.\using GCC with MinGW.assets\mouse-hover.png)

### C/C++ configurations

If we want more control over the c/c++ extension, we can create a `c_cpp_properties.json` file, which will allow we to change settings such as the path to the compiler, include paths, c++ standard, and more

we can view the **C/C++ configuration UI** by running the command `C/C++: Edit Configurations UI` from the command Palette (`ctrl+shift+p`)

![Command Palette](.\using GCC with MinGW.assets\command-palette.png)

This opens the `C/C++ Configurations` page. When we make changes here, VS Code writes them to a file called `c_cpp_properties.json` in the `.vscode` folder.

Here, we've changed the **Configuration name** to **GCC**, set the **Compiler path** dropdown to the g++ compiler, and the **IntelliSense mode** to match the compiler (gcc-x64)

 ![Command Palette](.\using GCC with MinGW.assets\intellisense-configurations-mingw.png)

Visual Studio Code places these setting in `.vscode\c_cpp_properties.json`. If we open that file directly, it should look something like this;

```json
{
  "configurations": [
    {
      "name": "GCC",
      "includePath": ["${workspaceFolder}/**"],
      "defines": ["_DEBUG", "UNICODE", "_UNICODE"],
      "windowsSdkVersion": "10.0.18362.0",
      "compilerPath": "C:/msys64/mingw64/bin/g++.exe",
      "cStandard": "c17",
      "cppStandard": "c++17",
      "intelliSenseMode": "windows-gcc-x64"
    }
  ],
  "version": 4
}
```

### Compiler path

The extension uses the `compilerPath` setting to infer the path to the infer the path to the C++ standard library header files. When the extension knows where to find those files, it can provide feature like smart completions and **Go to Definition** navigation.

The C/C++ extension attempts to populate `compilerPath` with the default compiler location based on what it finds on our system. The extension looks in several common compiler locations.

the `compilerPath` search order is:

- First check for the Microsoft Visual C++ compiler
- Then look for g++ on Windows Subsystem for Linux (WSL)
- Then g++ for Mingw-w64

If we have Visual studio or WSL installed, we may need to change `complerPath` to match the preferred compiler for our project. For example, if we install Mingw-w64 version8.1.0 using the i686 architecture, Win32 threading, and sjlj exception handing install options, the path would look like this: `C:\Program Files (x86)\mingw-w64\i686-8.1.0-win32-sjlj-rt_v6-rev0\mingw64\bin\g++.exe`



## reference

- [Using GCC with MinGW](https://code.visualstudio.com/docs/cpp/config-mingw)