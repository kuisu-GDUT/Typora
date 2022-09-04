# Viusal Studio IDE use

![img](.\VisualStudioIEDtoUse.assets\ide-overview.png)

## 创建程序

Dive in and create a simple program

1. start visual studio. The shart window appears with opeions for cloing a rep, opening a recent project, or creating a new project
2. choose create a new project

![Screenshot of the Visual Studio start menu with Create a new project selected.](.\VisualStudioIEDtoUse.assets\create-new-project.png)

The **create a new project** window opens and shows several project templates. A template contains the base files and settings required for a given project type.

3. To find a template, we can type or enter keywords in the search box. The list of available templates filters based on keywords we enter. we can further filter the template results by choosing C# from the All languages dropdown list, **Windows** from the All platforms list, and **Console** from the All project types list

   Select the Console Application template, and then select Next.

   ![Screenshot of the Create a new project window with Console Application selected.](.\VisualStudioIEDtoUse.assets\start-window-create-new-project.png)

4. In the **Configure your new project** windows, enter HelloWorld in the Project name box. Optionally, change the project directory location from the default location of "c:\users\<name>\source\repos", and then select next.

   ![Screenshot of the Configure your new project window with the project name HelloWorld entered.](.\VisualStudioIEDtoUse.assets\configure-new-project.png)

5. In the **Additional information** window, verify that .NET 6.0 appears in the target Framwork drop-down menu, and then select create.

   ![Screenshot of the Additional information window with .NET 6.0 selected.](.\VisualStudioIEDtoUse.assets\create-project-additional-info.png)

   Visual Studio creates the project. The program is is a simple "helloWorld" application that calls the `Console.WriteLine()` method to display the string "Hello, world!" in a console window

   The project files appear on the right side of the visual studio IDE, in a window called the Solution Explorer. In the **Solution Explorer** window, select the **Program.cs** file. The C# code for your app opens in the central editor window, which takes up most of the space.

   ![Screenshot that shows the Visual Studio IDE with the Program.cs code in the editor.](.\VisualStudioIEDtoUse.assets\overview-ide-console-app.png)

   The code is automatically colorized to indicate different parts, such as keywords and types. Line numbers help you locate code.

   Small, vertical dashed lines in the code indicate which braces match one another. We can also choose samll, boxed minus or plus signs to collapse  or expand blocks of code. This code outlining feature lets you hide code you don't need to see, helping to minimize onscreen clutter.

   ![Screenshot that shows the Visual Studio IDE with red boxes.](.\VisualStudioIEDtoUse.assets\overview-ide-console-app-red-boxes.png)

   Many other menus and tool windows are available.

6. Start the app by choosing `Debug > Start Without Debugging` from the Visual Studio top menu. or press **Ctrl + F5**

   ![Screenshot that shows the Debug > Start without Debugging menu item.](.\VisualStudioIEDtoUse.assets\overview-start-without-debugging.png)

   Visual Studio builds the app, and a console window opens with the message "Hello world!" we now have a running app

   ![Screenshot of the Debug Console window showing the output Hello, World! and Press any key to close this window.](.\VisualStudioIEDtoUse.assets\overview-console-window.png)

7. To close the console window, press any key.

8. Let's add more code to the app. Add the following C# code before the line that says `COnsole.WriteLine("Hello world!");`:

   ```C#
   Console.WriteLine("\nWhat is your name?");
   var name = Console.ReadLine();
   ```

   This code displays "What is you name?" in the console window, and then waits until the user enters some text.

9. Change the line that says `Console.WriteLine("Hello World)"; `to the following line:

   ```c#
   Console.WriteLine($"\nHello {name}!");
   ```

10. Run the app again by selecting `Debug > Start Without Debugging` or pressing Ctrl+5

    Visual Studio rebuilds the app, and a console window opens and prompts you for your name

11. Type your name in the console window and press Enter

    ![Screenshot of the Debug Console window showing the prompt for a name, the input, and the output Hello Georgette!.](.\VisualStudioIEDtoUse.assets\overview-console-input.png)

12. Press any key to close the console window and stop the running program.

## Use refactoring and IntelliSense

Let's look at a couple of the ways that `refactoring` and `IntelliSense` can help you code more efficiently

First, rename the `name` variable

1. Double-click the `name` variable, and type the new name for the variable, *useranme*

   A box appears around the variable, and a light bulb appears in the margin.

2. Select the light bulb icon to show the available `Quick Actions`. Select **Rename** 'name' to 'username'

   ![Screenshot that shows the Rename action in Visual Studio.](.\VisualStudioIEDtoUse.assets\rename-quick-action.png)

   The variable is renamed across the project, which in our case is only two places.

3. Now take a look at IntelliSense. Below the line that says: `Console.WriteLine($"\nHello {username}!");`, type `DateTime now = DateTime.`

   A box displays the members of the `DateTime` class. The description of the currently selected member also displays in a separate box.

   ![Screenshot that shows IntelliSense list members in Visual Studio.](.\VisualStudioIEDtoUse.assets\intellisense-list-members.png)

4. Select the member named **Now**, which is a property of the class, by double-clicking it or pressing Tab. Complete the line of code by adding a semicolon to the end of the line:`DateTime now = DateTime.Now;`.

5. Below that line, enter the following lines of code:

   ```c#
   int dayOfYear = now.DayOfYear;
   
   Console.Write("Day of year: ");
   Console.WriteLine(dayOfYear);
   ```

6. Next, use refactoring again to make the code a little more concise. Select the variable `now` in the line `DateTime now = DateTime.Now;`. A screwdriver icon appears in the margin on that line.

7. Select the screwdriver icon to see available suggestions from visual studio. This case shows the `Inline temporary variable` refactoring to remove a line of code without changing the overall code behavior

   ![Screenshot showing the Inline temporary variable suggestion in Visual Studio.](.\VisualStudioIEDtoUse.assets\inline-temporary-variable-refactoring.png)

8. Select **Inline temporary variable** to refactor the code .

9. Run the program again by pressing Ctrl+5. The output looks something like this:

   ![Screenshot of the Debug Console window showing the prompt for a name, the input, and the output 'Hello Georgette! Day of year: 244'.](.\VisualStudioIEDtoUse.assets\overview-console-final.png)

### Debug code

When you write code, you should run it and test it for bugs. Visual Studio debugging system lets you step through code one statement at a time and inspect variables as you go. You can *breakpoints* that stop execution of the code at a particular line, and observe how the variable value changes as the code runs.

Set a breakpoint to see the value of the `username` variable while the program is running.

1. Set a breakpoint on the line of code that sys `Console.Write($"n\Hello {username}!");` by clicking in the far-left margin, or gutter, next to the line. You can also select the line of code and then press F9

   A red circle appears in the gutter, and the line is highlighted.

   ![Screenshot that shows a breakpoint on a line of code in Visual Studio.](.\VisualStudioIEDtoUse.assets\breakpoint.png)

2. Start debugging by selecting `Debug > Start Debugging` or pressing F5.

3. When the console window appears and asks for your name, enter your name.

   The focus returns to the Visual Studio code editor, and the line of code with the breakpoint is highlighted in yellow. The yellow highlight means that this line of code will execute next. The breakpoint makes the app pause execution at this line

4. Hover your name over the `username` variable to see its value. You can also right-click on `username` and select **Add Watch** to add the variable to the Watch window, where you can also see its value

   ![Screenshot that shows a variable value during debugging in Visual Studio.](.\VisualStudioIEDtoUse.assets\debugging-variable-value.png)

5. Press F5 again to finish running the app.

## Customize Visual Studio

you can personalize the visual studio user interface, including changing the default color theme. To change the color theme:

1. On the menu bar, choose **Tool > Options** to open the Options dialog.

2. On the **Environment > General** options page, change the **Color Theme** selection to **Blue** or **Light**, and then select OK

   the color theme for entire IDE change according. The following screenshot show the Blue theme

   ![Screenshot that shows Visual Studio in Blue theme.](.\VisualStudioIEDtoUse.assets\blue-theme.png)

# Create a console calculator in C++

The usual stating point for a C++ programmer is a "hello, world" application that runs on the command line. That's what you'll create first in visual studio in this article, and then we'll move on to something more challenging: a calculator app.

## Create app project

A project contains all the options, configurations, and rules used to build your apps. It also manages the relationship between all the project's files and any external files. To create you app first, you'll create a new project and solution

1. if you've just started visual studio, you'll see the visual studio start dialog box. choose **Create a new project** to get started.

   ![Screenshot of the Visual Studio 2022 initial dialog.](.\VisualStudioIEDtoUse.assets\calc-vs2022-initial-dialog.png)

   Otherwise, on the menu bar in Visual studio, choose **File > New > Project**. The **Create a new project** window opens.

2. In the list of project templates, choose **Console App**, then choose Next.

   ![Screenshot of choosing the Console App template.](.\VisualStudioIEDtoUse.assets\calc-vs2019-choose-console-app.png)

3. In teh **Configure your new project** dialog box, select the **Project name** edit box, name your new project "CalculatorTutorial", then choose Create.

   ![Name your project in the Configure your new project dialog.](.\VisualStudioIEDtoUse.assets\calc-vs2019-name-your-project.png)

   An empty C++ Windows console application gets created. Console applications use a Windows console window to display output and accept user input. In Visual Studio, an editor window opens and shows the generated code

   ```c++
   // CalculatorTutorial.cpp : This file contains the 'main' function. Program execution begins and ends there.
   //
   
   #include <iostream>
   
   int main()
   {
       std::cout << "Hello World!\n";
   }
   
   // Run program: Ctrl + F5 or Debug > Start Without Debugging menu
   // Debug program: F5 or Debug > Start Debugging menu
   
   // Tips for Getting Started:
   //   1. Use the Solution Explorer window to add/manage files
   //   2. Use the Team Explorer window to connect to source control
   //   3. Use the Output window to see build output and other messages
   //   4. Use the Error List window to view errors
   //   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
   //   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
   ```

## Verify that your new app builds and runs

The template for a new Windows console application creates a simple C++ "Hello world" app. At this point, you can see how Visual Studio builds and runs the apps you create right from the IDE

1. TO build your project, choose **Build Solution** from the **Build** menu. The **Output** window shows the results of the build process.

   ![Screenshot of Visual Studio 2019 with the Output window showing the result of the build process.](.\VisualStudioIEDtoUse.assets\calc-vs2019-build-your-project.png)

2. To run the code, on the menu bar, choose **Debug > Start without debugging**

   ![Screenshot of the Visual Studio 2019 Microsoft Visual Studio Debug Console showing the code ran successfully.](.\VisualStudioIEDtoUse.assets\calc-vs2019-hello-world-console.png)

## Edit the code

Now let's turn the code in this template into a calculator app.

1. In the *CalculatorTutorial.cpp* file, edit the code to match this example

   ```C++
   // CalculatorTutorial.cpp : This file contains the 'main' function. Program execution begins and ends there.
   //
   
   #include <iostream>
   
   using namespace std;
   
   int main()
   {
       cout << "Calculator Console Application" << endl << endl;
       cout << "Please enter the operation to perform. Format: a+b | a-b | a*b | a/b"
           << endl;
       return 0;
   }
   ```

   > Understanding the code:
   >
   > - the `#include` statements allow you to reference code located in other files. Sometimes, you may see a filename surrounded by angle brackets (<>); Other times, it;s surrounded by quotes (""). In general, angle brackets are used when referencing the C++ standard Library, while quotes are used for other file.
   > - The `using namespace std;` line tell the compiler to expect stuff from the C++ Standard Library to be used in this file. Without this line, each keyword from the library would have to be preceded with a `std::`, to denote its scope. For instance, without that, each reference to `cout` would have to written as `std::cout`. The `using` statement is added to make the code look more clean.
   > - the `cout` keywords is used to print to standard output in C++. The << operator tells the compiler to send whatever is to the right of it to the standard output.
   > - The **endl** keywords is like the Enter key; it ends the line and moves the cursor to the next line. It is a better practice to put a `\n` inside the the string (contained by "") to do the same thing, as `endl` always flushes the buffer and can hurt the performance of the program, but since this is a very small app, `endl` is used instead for better readability.
   > - All C++ statements must end with semicolons and all C++ applications must contain a main() function. This function is what the program runs at the start. All code must be accessible from `main()` in order to be used.

2. To save the file, enter Ctrl+s the floppy disk icon in the toolbar under the menu bar.

3. To run the application, press Ctrl+F5 or go to Debug menu and choose **Start Without Debugging**. you should see a console window appear that displays the text specified in the code.

4. Close the console window when you're done.

## Add code to do some match

It's time to add some math logic.

### To add a Calculator class

1. Go to the project menu and choose Add class. In the Class name edit box, enter Calculator. Choose ok. Two new files get added to your project. To save all your changed files at once, press ctrl+shift+s. It's a keyboard shortcut for **File  > Save all**. There's also a toolbar button for save all, an icon of two floppy disks, found beside the save button. In general, It's good practice to do save all frequently, so you don't miss any files when you save.

   ![Screenshot of the Add Class dialog box with Calculator typed in the Class Name text box.](.\VisualStudioIEDtoUse.assets\calc-vs2019-create-calculator-class.png)

   A class is like a blueprint for an object that does something. In this case, we define a calulator and how it should work. The **Add class** wizard you used above create .h and .cpp files that have the same name as the class. You can see a full list of your project files in the **Solution Explorer** window, visible on the side of the IDE. If the window isn't visible, you can open it from the menu bar: choose **View > Solution Explorer**

   ![Screenshot of the Visual Studio 2019 Solution Explorer window displaying the Calculator Tutorial project.](.\VisualStudioIEDtoUse.assets\calc-vs2019-solution-explorer.png)

   You should now have three tabs open in the editor: *CalculatorTutorial.cpp, Calculator.h, and Calculator.cpp*. If you accidentally close one of them, you can reopen it by double-clicking it in the **Solution Explorer** window.

2. In **Calculator.h**, remove the `Calculator();`, and `~Calculator();` lines that were generated, since you won't need them here. Next, add the following line of code so the file now looks like this:

   ```c++
   #pragma once
   class Calculator
   {
   public:
       double Calculate(double x, char oper, double y);
   }
   ```

   > Understanding the code
   >
   > - The line you added declares a new function called `Calculate`, which we'll used to run match operations for addition, subtraction, multiplicatioin, and division.
   > - C++ code is organized into *header (.h)* files and source (.cpp) files. Several other file extensions are supported by various compilers, but these are the main ones to know about. Functions and variables are normally *declared*, that is, given a name and a type, in header files, and *implemented*, or given a definition, in source files. To access code defined in another file, you can use `#include "filename.h"`, where 'filename.h' is the name of the file that declares the variables or functions you want to use.
   > - The two lines you deleted declared a *constructor and destructor* for the class. For a simple class like this one, the compiler creates them for you, and their uses are beyond the scope of this tutotial
   > - It's good practice to organize your code into different files based on what it does, so it's easy to find the code you need later. In our case, we define the `Calculator` class separately from the file containing the `main()` function, but we plan to reference the `Calculator` class in `main()`

3. You'll see a green squiggle appear under `Calculate`. It's because we haven't defined the `Calculate` function in the .cpp file. Hover over the world, click the lightbulb (in this case, a screwdriver) that pops up, choose **Create definition of 'Calculate' in Calculator.cpp**

   ![Screenshot of Visual Studio 2019 showing the Create definition of Calculate in Calculator C P P option highlighted.](.\VisualStudioIEDtoUse.assets\calc-vs2019-create-definition.png)

   A pop-up appears that gives you a peek of the code change that was made in the other file. The code was added to *Calculator.cpp*

   ![Pop-up with definition of Calculate.](.\VisualStudioIEDtoUse.assets\calc-vs2019-pop-up-definition.png)

   Currently, it just returns 0.0 Let's change that, Press Esc to close the pop-up.

4. Switch to the `Calculator.cpp` file in the editor window. Remove the `Calculator()` and `~Calculator()` sections (as you did in the .h file) and add the following code to `Calculate()`:

   ```c++
   #include "Calculator.h"
   
   double Calculator::Calculate(double x, char oper, double y)
   {
       switch(oper)
       {
           case "+":
               return x+y;
           case "-":
               return x-y;
           case "*":
               retrun x*y;
           case "/":
               return x/y;
           default:
               return 0.0;
       }
   }
   ```

   > Understanding the code
   >
   > - The function `Calculate` consumes a number, an operator, and a second number, then performs the requested operation on the numbers.
   > - The switch statement checks which operator was provided, and only executes the case corresponding to that operation. The default: case is a  fallback in case the user types an operator that isn't accepted, so the program doesn't break. in general, it's best to handle invalid user input in a more elegant way, but this is beyond the scope of the tutorial
   > - the `double` keyword denotes a type of number that supports decimals. This way, the calculator can handle both decimal math and integer math. The `Calculate` function is required to always return such a number due to the `double` at the very start of the code (this denotes the function's return type), which is why we return 0.0 even in the default case.
   > - The .h file declares the functions `prototype`, which tell the compiler upfront what parameters it requires, and what return type to expect from it. The .cpp file has all the implementation details of the function.

   If you build and run the code again at this point, it will still exit after asking which operation to perform. Next, you'll modify the `main` function to do some calculations.

### To call the Calculator class member functions

1. now let's update the `main` function in *CalculatorTutorial.cpp*:

   ```c++
   // CalculatorTutorial.cpp : This file contains the 'main' function. Program execution begins and ends there.
   //
   
   #include <iostream>
   #include "Calculator.h"
   
   using namespace std;
   
   int main()
   {
       double x = 0.0;
       double y = 0.0;
       double result = 0.0;
       char oper = '+';
   
       cout << "Calculator Console Application" << endl << endl;
       cout << "Please enter the operation to perform. Format: a+b | a-b | a*b | a/b"
            << endl;
   
       Calculator c;
       while (true)
       {
           cin >> x >> oper >> y;
           result = c.Calculate(x, oper, y);
           cout << "Result is: " << result << endl;
       }
   
       return 0;
   }
   ```

   > Understanding the code
   >
   > - Since C++ programs always start at the `main()` function, we need to call our other code from there, so a `#include` statement is needed.
   > - Some initial variables `x,y,oper, result` are declared to store the first number, second number, operator, and final result, respectively. It is always good practice to give them some initial values to avoid undefined behavior, which is what is done here. 
   > - The `Calculator c;` line declares an object named "c" as an instance of the `Calculator` class. The class itself if just a blueprint for how calculators work; the object is the specific calculator that does the math.
   > - The `while (true)` statement is a loop. The code inside the loop continues to execute over and over again as long as the condition inside the `()` holds true. Since the condition is simply listed as `true`, it's always true, so the loop runs forever. To close the program, the user must manually close the console window. Ohterwise, the program always waits for new input.
   > - The `cin` keyword is used to accept input from the user. This input stream is smart enough to process a line of text entered in the console window and place it inside each of the variables listed, in order, assuming the user input matches the required specification . you can modify this line to accept different types of input, for instance, more than two number, though the `Calculate()` function would also need to be updated to handle this.
   > - The `c.Calculate(x,oper,y);` expression calls the `Calculate` function defined earlier, and supplies the entered input values. The function then returns a number that gets stored in `result`.
   > - Finally, `result` is printed to the console, so the user sees the result of the calculation.

### Build and test the code again

Now it's time to test the program again to make sure everything works properly.

1. Press ctrl+F5 to rebuild and start the app
2. Enter `5+5`, and press Enter. Verify that the result is 10.
   ![Screenshot of the Visual Studio 2019 Microsoft Visual Studio Debug Console showing the correct result of 5 + 5.](E:\kuisu\typora\VisualStudioIDE\VisualStudioIEDtoUse.assets\calc-vs2019-five-plus-five.png)

## 问题

### visual studio 调试时提示 已加载“C:\Windows\SysWOW64\ntdll.dll”。无法查找或打开 PDB 文件

- 解决方式

点调试, 然后选项和设置

![这里写图片描述](E:\kuisu\typora\VisualStudioIDE\VisualStudioIEDtoUse.assets\20160106124459573)

右边勾上启用源服务器支持

![这里写图片描述](E:\kuisu\typora\VisualStudioIDE\VisualStudioIEDtoUse.assets\20160106124523540)

左边点符号, 把微软符号服务器勾选上

![这里写图片描述](E:\kuisu\typora\VisualStudioIDE\VisualStudioIEDtoUse.assets\20160106124531645)

接下来, 运行时等一下, 加载完成之后就可以

只是第一次加载, 加载完成以后, 可以把之前勾选的取消掉.

### OpenCV中出现“Microsoft C++ 异常: cv::Exception，位于内存位置 0x0000005C8ECFFA80 处。”的异常

问题:

```c++
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;
 
int main()
{
	Mat image = imread("D:\Test\2.jpg");  //存放自己图像的路径 
	imshow("显示图像", image);
	waitKey(0);
	return 0;
}
```



![img](E:\kuisu\typora\VisualStudioIDE\VisualStudioIEDtoUse.assets\20180328143341952)

原因: 主要时图片文件是用Windows文件资源管理器里面复制古来的.

解决方案: 将含有文件路径的单斜杠"\\"改变成两个斜杠"\\\\\"

### 关于opencv报错：未定义标识符"CV_WINDOW_AUTOSIZE"

报错: 未定义标识符"CV_WINDOW_AUTOSIZE"

解决方案: 在代码开头加入头文件#include <opencv2/highgui/highgui_c.h>

## reference

- [Create a console calculator in C++](https://docs.microsoft.com/en-us/cpp/get-started/tutorial-console-cpp?view=msvc-170&viewFallbackFrom=vs-2022)