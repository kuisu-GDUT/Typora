# Python in Visual Studio Code

Working with python in Visual Studio Code, using the Microsoft python extension, is simple, fun, and productive. The extension makes VS Code an excellent Python editor, and works on any operating system with variety of Python interpreters. It leverages all of VS Codes power to provide auto complete and InterlliSense, linting, debugging, and unit testing, along with the ability to easily switch between Python environments, including virtual and conda environments. 

This article provides only an overview of the different capabilities of the python extension for VS Code. For a walkthrough of editing, running, and debugging code, use the button below.

## Install Python and the Python extension

This tutorial introduces you to VS Code as a python environment, primarily how to edit, run, and debug code through the following tasks:

- Writ, run, and debug a Python "hello world" Application
- Learn how to install package by creating python virtual environments
- write a simple python script to plot figures within vs Code

### Prerequisites

1. VS Code
2. VS Code Python extension
3. Python3

### Install Python extension

install the Python extension for VS Code from the visual studio marketplace. For  additional details on installing extensions, see Extension Marketplace. The Python extension is named Python and it's published by Microsoft.

![Python extension on Marketplace](.\Python in VS code.assets\python-extension-marketplace.png)

### Install a Python interpreter

Along with the python extension, you need to install a Python interpreter. Which interpreter you use is dependent on your specific needs.

Install [Python from python.org](https://www.python.org/downloads/). You can typically use the Download Python button that appears first on the page to download the latest version/

### Verify the python installation

To verify that you have installed Python successfully on you machine, run one of the following commands

```cmd
py -3 --version
```

### start VS Code in a project folder

Using a command prompt or terminal, create an empty folder called "hello", navigate into it, and open VS code in that folder by entering the following commands.

```shell
mkdir hello
cd hello
code.
```

By starting VS Code in a folder, that folder becomes you "workspace". VS Code stores setting that are specific to that workspace in `./vscode/setting.json`, which are separate from user setting that are stored globally.

Alternately, you can run VS Code through the operating system UI, then use `File > Open Folder` to open the project folder.

### Select a Python interpreter

Python is an interpreted language, and in order to run Python code and get Python IntelliSense, you must tell VS Code which interpreter to use

From within VS Code, select a Python 3 interpreter opening the Command Palette(`ctrl+shift+p`), start typing the `Python: Select interpreter` command to search, then select the command. You can also use the Select Python Environment option on the Status Bar if available (it may already show a selected interpreter, too)

![No interpreter selected](.\Python in VS code.assets\no-interpreter-selected-statusbar.png)

The command presents a list of available interpreter that VS Code can find automatically, including virtual environments. 

Selecting an interpreter sets which interpreter will be used by the Python extension for that worksapce.

### Create a Python Hello World source code file

From the File Explorer toolbar, select the New File button on the `hello` folder.

![File Explorer New File](.\Python in VS code.assets\toolbar-new-file.png)

Name the file `hello.py`, and it automatically opens in the editor:

![File Explorer hello.py](.\Python in VS code.assets\hello-py-file-created.png)

By using the `.py` file extension, you tell VS Code to interpret this file as Python program, so that it evaluates the contents with python extension and the select interpreter.

Now that you have a code file in your workspace, enter the following source code in `hello.py`

```python
msg = 'hello world'
print(msg)
```

When you start typing `print`, notice how `intelliSense` presents auto-completion options.

![IntelliSense appearing for Python code](.\Python in VS code.assets\intellisense01.png)

IntelliSense and auto-completions work for standard Python modules as well as other package you've installed into the environment of the select Python interpreter. It also provides completions for methods available on object types. For example, because the `msg` veriable contains a string, InterlliSense provides string methods when you type `msg`

![IntelliSense appearing for a variable whose type provides methods](.\Python in VS code.assets\intellisense02.png)

Fell free to experiment with IntelliSense some more, but then revert your changes so you have only the `msg` variable and the `print` call, and save the file (ctrl+s)

### Run hello world

It's simple to run `hello.py` with Python. Just click the Run Python File in Terminal play button in the top-right side of the deitor

![Using the run python file in terminal button](.\Python in VS code.assets\run-python-file-in-terminal-button.png)

the Button opens a terminal panel in which your python interpreter is automatically activated, then run `python hello.py`

![Program output in a Python terminal](.\Python in VS code.assets\output-in-terminal.png)

There are three other ways you can run python code within vs code

- right-click anywhere in the editor window and select run python file in termial

![Run Python File in Terminal command in the Python editor](.\Python in VS code.assets\run-python-file-in-terminal.png)

- select one or more lines, then press `shift+enter` or right-click and select run selection/Line in python terminal. This command is convenient for testing just a part of a file
- From the command Palette (ctrl+shift+P), select the `Python:start REPL` command to open a REPL terminal for the currently selected Python interpreter. In the REPL, you can then enter and run lines of code at a time

### Configure and run the debugger

First, set a breakpoint on line 2 of `hello.py` by placing the cursor on the `print` call and pressing `F9`. Alternately, just click in the editor's left gutter, next to the line numbers. when you set a breakpoint, a red circle appears in the gutter.

![Setting a breakpoint in hello.py](.\Python in VS code.assets\breakpoint-set.png)

Next, to initialize the debugger, press `F5`. Since this is you first time debugging this file, a configuration menu will open from the command palette allowing you to select the type of debug configuration you would like for the opened file

![Debug configurations after launch.json is created](.\Python in VS code.assets\debug-configurations.png)

Note: VS Code uses JSON file for all of it various configurations; `launch.json` is the standard name for a file containing debugging configurations.

These different configurations are fully explained in Debugging configurations; for now, just select Python file, which is the configuration that runs the current file shown in the editor using the currently selected Python interpreter.

You can also start the debugger by clicking on the down-arrow next to the run button on the editor, and selecting Debug python File in Termial

![Using the debug Python file in terminal button](.\Python in VS code.assets\debug-python-file-in-terminal-button.png)

The debugger will stop at the first line of the file breakpoint. The current line is indicated with a yellow arrow in the left margin. If you examine the local variables window at this point, you will see now defined `msg` variable appears in the local pane.

![Debugging step 2 - variable defined](.\Python in VS code.assets\debug-step-02.png)

A debug toolbar appears along the top with the following commands from left to right. continue(f5), step over(f10), step into(f11), step out (shift+f11), restart (ctrl+shift+F5), and stop (shift+F5).

![Debugging toolbar](.\Python in VS code.assets\debug-toolbar.png)

The status bar also change color to indicate that you're in debug mode. The Python Debug Console also appears automatically in the lower right panel to show the commands being run, along with the program output.

You can also work with variables in the Debug Console. Then try entering the following lines, one by one, at . prompt at the console.

```python
msg
msg.capitalize()
msg.split()
```

![Debugging step 3 - using the debug console](.\Python in VS code.assets\debug-step-03.png)

### Install and use package

Lets now run an example that's a little more interesting. In python, packages are how you obtain any number of useful code libraries, typically from [PyPI](https://pypi.org/). For this example, you use the `matplotlib` and `numpy` packages to create a graphical plot as is commonly done with data science. 

Return to **Explorer** view, create a new file called `standaradplot.py`, and paste in the following source code

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 20, 100)  # Create a list of evenly-spaced numbers over the range
plt.plot(x, np.sin(x))       # Plot the sine of each x point
plt.show()                   # Display the plot           # Display the plot
```

Next, try running the file in the debugger using the "python: Current file" configuration as described in the last section.

A best practice among Python developers is to avoid installing packages into a global interpreter environment. You install use a project-specific `virtual environment` that contains a copy of a global interpreter. Once you activate that environment, any packages you then install are isolated from other environments. Such isolation reduces many complications that can arise from conflicting package version. To create a *virtual environment* and install the required packages, enter the following commands as appropriate for you operating system:

1. Create and activate the virtual environment

   > Note: When you create a new virtual environment, you should be prompted by VS Code to set it as the default for your workspace folder. if selected, the environment will automatically be activated when you open a new terminal

   ![Virtual environment dialog](.\Python in VS code.assets\virtual-env-dialog.png)

   ```cmd
   py -3 -m venv .venv
   ./venv/scripts/activate
   ```

   If the activate command generates the message "Activate.ps1 is not digitally signed. You cannot run this script on the current system", then you need to temporarily change the powershell execution policy to allow script to run

   ```cmd
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
   ```

2. select you new environment by using the `Python: Select interpreter` command from the command palette.

3. Install the packages

   ```cmd
   python -m pip install matplotlib
   ```

4. Rerun the program now (with or without the debugger) and after a few moments a plot window appears with output

![matplotlib output](.\Python in VS code.assets\plot-output.png)

5. Once you are finished, type `deactivate` in the terminal window to deactivate the virtual environment.
6. \



## reference

- [getting started with python in vs code](https://code.visualstudio.com/docs/python/python-tutorial)