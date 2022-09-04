# Windows下的C++编译工具

MinGW是Windows系统下的一个编译环境, 包含了C++代码编译所需的三方库, 头文件等, 用于完成C++源码编译和链接

- 安装

  利用msys2下载安装MinGW

- 配置

  根据安装目录的bin目录配置到系统环境变量中的Path, 通过`g++ --version`验证是否安装成功

- 测试编译

  编写一个C++代码文件

  ```c++
  #include <iostream>
  using namespace std;
  
  int main() 
  {
      cout << "Hello, World!";
      return 0;
  }
  ```

  先编译但不链接, 然后链接生成可执行文件

  ```cmd
  $ g++ -C helloworld.cpp
  $ g++ helloworld.o -o helloworld.exe
  ```

  假如使用`-c`标识, 则是编译但不链接, 会生成对应的`.o`文件, 不设置标识, 则会直接生成最终的`.exe`可执行文件.

- 静态关联

  上述编译生成的可执行文件, 双击打开都会提示"无法启动此程序, 因为计算中丢失libgcc_s_dw-1.dll'. 这样的错误, 这是因为gcc编译器编译时默认使用的不是静态关联的方式, 运行时找不到对应的库导致报错

  解决方案: 可以直接将确实的dll库手动复制到可执行文件目录下. 或者在链接时带上标识-static-libgcc来指定静态关联即可

  ```c++
  g++ helloworld.o -o hellowrld.exe -static-libgcc
  ```

## make

> 1. gcc是用于编译和链接工具, 编译少量文件可以直接使用gcc命令完成, 但党源文件很多, 用gcc命令去逐个编译则是很混乱的
> 2. 通过make来管理编译过程, make安装makefile中的命令进行编译和链接, 而makefile命令中就包含了调用gcc去编译某个源文件的命令

- MinGW配置make

  MinGW-w64集成了gcc和make(gcc.exe和bin/mingw32-make.exe)的工具, 上我们已经分别配置好了MinGW-w64, 在通过以下配置进行完成make编译

  1. 打开MinGW-w64的bin目录, 拷贝一份mingw32-make.exe改名为make.exe

  ![image-20211117185900660](.\Windows下的C++编译工具.assets\image-20211117185900660-16371467414141.png)

- 优化

  在window下需要输入mingw32-make.exe确实有些别扭. 这里使用简单粗暴的方法, 直接[复制]-->[粘贴]然后重命名make.exe