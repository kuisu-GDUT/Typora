# 最新的Jupyter Notebook可以混合执行Shell、Python以及Ruby、R等代码！

## Magics

主要由两种语法

1. Line magics: `%`字符开始, 该行后面都为指令代码, 参数用空格隔开,不需要加引号

2. Cell magics: 使用两个百分号`%%`开始,后面的整个单元(cell) 都是指令代码.

   注: `%%`魔法操作只在cell的第一行使用, 且不能嵌套,重复.

- 输入`%lsmagic`获得Magic操作符列表

  ```python
  %lsmagic
  ```

  缺省情况下, `Automagic`开关打开, 不需要输入`%`符号, 将会自动识别. 注意, 这由可能与其他的操作引起冲突. 

- 执行shell脚本

  ```jupyter
  ls -l -h
  ```

- 执行多行shell脚本

  ```jupyter
  %%!
  ls -l
  pwd
  who
  ```

