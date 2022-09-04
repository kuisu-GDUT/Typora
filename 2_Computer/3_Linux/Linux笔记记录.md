Linux

## 集群使用

### 集群相关信息

-   登录节点

    `192.168.61.6`

-   计算节点

    `sp: 10.53.2.205`

    `sm2: 10.53.4.142`

    `sm1: 10.53.2.197`

-   软件下载节点

    `gsw: 10.53.11.51 (cngb-software-0-1)`

-   windows-2060s

    `172.16.217.51`

-   ubuntu-18-A5000

    `172.16.18.21`

    `anaconda: /home/implab5/anaconda3/bin`

-   ubuntu-20-A5000

    `172.16.18.22`

    `anaconda: /home/implab6/anaconda3/etc/profile.d/conda.sh`

-   anconda 安装路径

    `/hwfssz1/ST_BIOINTEL/P20Z10200N0039/07.software/anaconda3`

    在个人的`./bashrc`文件末尾添加以下内容

    `. /hwfssz1/ST_BIOINTEL/P20Z10200N0039/07.software/anaconda3/etc/profile.d/conda.sh`

    -   conda环境配置

        可以在个人账号下进入A100计算节点, 输入: `conda config --set proxy_servers.http http://10.53.11.51.:5896`

-   bash

    ```bash
    alias idb="cd /zfssz3/ST_BIGDATA/stereomics"
    alias gpu1="ssh sukui@cngb-gpu-e16-1"
    alias a100="ssh 10.53.3.163"
    alias gpu2="ssh sukui@cngb-gpu-e16-2"
    alias 50t="cd /hwfssz1/ST_BIOINTEL/P20Z10200N0039/06.user/sukui"
    alias 100t="cd /jdfssz2/ST_BIOINTEL/P20Z10200N0039/06.user/sukui"
    alias sc="ssh $USER@10.54.25.108"
    alias sp="ssh $USER@10.53.2.205" # bcbuild
    alias sp2="ssh $USER@10.54.0.106"
    alias sgc="ssh $USER@10.53.7.60" # writefq compile server
    alias sm1="ssh $USER@10.53.2.197" #cngb-supermem-f19-7, large memory node
    #alias sm2="recordPWD && ssh $USER@10.53.2.198" #cngb-supermem-f19-8, large memory node
    alias sm2="ssh $USER@10.53.4.142" #cngb-supermem-a14-2, large memory node
    alias gsw="ssh $USER@10.53.11.51"  # cngb-software-0-1
    ```

-    注意事项

    ```bash
    # 台式电脑ip
    # 172.16.229.202
    
    #
    ```

    

-   时空组目录

    公共路径, 以及存储用于代码, 工具,,软件和测试数据, 二级存储用于大量数据存储

    | **目录**         | **路径**                            | **容量** |
    | ---------------- | ----------------------------------- | -------- |
    | 时空组学一级存储 | /hwfssz1/ST_BIOINTEL/P20Z10200N0039 | 50T      |
    | 时空组学二级存储 | /jdfssz2/ST_BIOINTEL/P20Z10200N0039 | 100T     |

      计算节点(出指定测试人员外, 组内成员默认只有A100节点权限)

    | **shortcut** | **ip**      | **alias**      | **CPU cores** | **Memory** | **GPU** |
    | ------------ | ----------- | -------------- | ------------- | ---------- | ------- |
    | gpu1         | 10.53.8.72  | cngb-gpu-e16-1 | 48            | 192G       | 4xP100  |
    | gpu2         | 10.53.8.73  | cngb-gpu-e16-2 | 48            | 192G       | 4xP100  |
    | a100         | 10.53.3.163 | cngb-gpu-m09-1 | 256           | 2T         | 8xA100  |

-   任务投递

    `qsub -cwd -l vf=30g, num_proc=12 -P P20Z10200n0039 -q st.q script.sh`

    >   `-cwd`: 使用当前目录作为工作目录
    >
    >   -l 请求资源, vf代表内存, num_proc指定线程数
    >
    >   -P 项目名称, 必须要是系统中存在的项目, 且用户必须为项目中成员
    >
    >   -q st.q: 任务队列, 一般任务投递均写st.q, GPU任务写st_gpu1.q, 投递到A100的任务写st_a100.q
    >
    >   script.sh 需要运行的脚本

    Example: `test.sh`
    ![image-20220509202759329](Linux笔记记录.assets/集群-任务投递.png)

    `test.py`
    ![image-20220509202850203](Linux笔记记录.assets/集群-任务投递2.png)
    任务投递成功可以用qstat查看,, 可以通过state查看当前状态

    ![image-20220509202950196](Linux笔记记录.assets/集群-任务投递3.png)

    `qstat -j jobID`可以查看详细信息

    ![image-20220509203042141](Linux笔记记录.assets/集群-任务投递4.png)

    -   异常处理

        投上去的任务由于存储访问的问题出现EQW状态, 需要手动清理, 不然任务会一直挂在上面

        ![image-20220509203235261](Linux笔记记录.assets/集群-任务投递5.png)

        `qmod -cj jobID`: 清楚Eqw状态

        `qdel jobID`:  直接杀掉任务

-   pycharm 连接集群

    ```bash
    ssh -NfL PRIVATE_PORT:TARGET_SERVER:22 USER_NAME@LOGIN_SERVERsss
    ssh -NfL 5002:10.53.4.142:22 sukui@192.168.61.6
    ```

成功建立隧道的示例

![image-20220803095119062](Linux笔记记录.assets/image-20220803095119062.png)

### 配置个人环境

-   软连接conda

    ```bash
    vi ~/.bashrc
    #add conda location
    . /home/xxx/anaconda3/etc/profile.d/conda.sh
    conda --version
    conda -info -e
    conda create --name py39 python=3.9 -y
    ```

-   创建相关隐藏目录及个人相关目录

    ```bash
    mkdir -p ~/.astropy ~/.bash_history ~/.bash_logout ~/.cache ~/.condarc ~/.config ~/.emacs ~/.pycharm_helpers ~/.python_history ~/.ssh ~/.viminfo ~/.zshrc
    
    mkdir -p ~/1_data ~/2_software ~/3_pearson ~/4_project ~/5_python ~/other
    ```

-   移动到挂载的存储路径下

    ```bash
    mv ~/.a* /media/Data/sukui/ #将.a开头的所有文件链接到目标路径下
    ```

-   构建软连接

    ```bash
    ln -s /media/Data/sukui/ ~/#将文件夹链接到目标路径
    ```

-   设置别名

    ```
    vi .bashrc
    alias bgi="cd /hwfssz1/ST_BIOINTEL/P20Z10200N0039/06.user/sukui"
    ```

    

-   构建pytorch相关环境

    ```bash
    nvidia-smi
    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y#cuda为11
    conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch#cduda为10
    conda install pytorch torchvision torchaudio cpuonly -c pytorch
    pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113#pip安装cuda11
    pip3 install torch torchvision torchaudio#pip安装cuda10
    ```

    -   download yolov5为测试环境

        ```bash
        git clone https://github.com/ultralytics/yolov5  # clone
        cd yolov5
        pip install -r requirements.txt  # install
        wget https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt#下载yolov5.pt
        python train.py --img 640 --batch 16 --epochs 3 --data coco128.yaml --weights yolov5s.pt
        ```

## conda

-   

-   创建环境

    `conda create --name py38 python=3.8`

-   conda无法更新或者安装: `raise Response304ContentUnchanged()``

    `conda clean -i`

-   查看conda的环境

    `conda info -e`

-   激活环境

    `conda activate py38`

-   pytorch 1.10安装

    `conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=10.2 -c pytorch`

-   离线安装

    找到之后就直接down下来了，下载到的是一个.tar.gz2的压缩文件，之后将压缩文件放在Anaconda安装目录的pkgs目录下（注：conda install 缓存文件路径为anaconda/pkgs）

    如果是因为网络原因某个包下载失败，可以删除原来下载的不完整的缓冲文件，再复制或本地安装即可，例如：

    ```text
    conda install --use-local pytorch-0.4.0-py35_cuda8.0.61_cudnn7.1.2_1.tar.bz2
    ```

    --use-local后面最好写绝对路径，或者进入到Anaconda的pkgs目录下再执行上述语句。

### jupyternotebook

pack

修改配置文件jupyter_notebook_config.py中的
c.NotebookApp.notebook_dir = ‘’
改为要修改的[根目录](https://so.csdn.net/so/search?q=根目录&spm=1001.2101.3001.7020)
把单引号换成双引号
c.NotebookApp.notebook_dir =“E:\学习\[jupyter](https://so.csdn.net/so/search?q=jupyter&spm=1001.2101.3001.7020)”

## Windows

-   开启ping

    `netsh firewall set icmpsetting 8`

-   安装ssh

    1.  首先卸载利用 Add feature 进行安装的 OpenSSH，并重启。

    2.  重启之后遵照微软的 PowerShell 的 github 的[教程](https://link.zhihu.com/?target=https%3A//github.com/PowerShell/Win32-OpenSSH/wiki/Install-Win32-OpenSSH)进行安装，小结如下。

    3.  1.  确认[注意事项](https://link.zhihu.com/?target=https%3A//github.com/PowerShell/Win32-OpenSSH/wiki/Various-Considerations)以及[项目的范围](https://link.zhihu.com/?target=https%3A//github.com/PowerShell/Win32-OpenSSH/wiki/Project-Scope)；

        2.  下载最新的 OpenSSH 安装文件 [[下载地址](https://link.zhihu.com/?target=https%3A//github.com/PowerShell/Win32-OpenSSH/releases/latest)]；

        3.  将下载的zip文件解压到 `C:\Program Files\OpenSSH`

        4.  用管理员权限打开 PowerShell， 进入安装目录，运行

            `powershell.exe -ExecutionPolicy Bypass -File install-sshd.ps1`

        5.  为 ssh 设置防火墙，运行指令

            `netsh advfirewall firewall add rule name=sshd dir=in action=allow protocol=TCP localport=22`

        6.  此时，我们回到微软的[官方OpenSSH安装文档](https://link.zhihu.com/?target=https%3A//docs.microsoft.com/en-us/windows-server/administration/openssh/openssh_install_firstuse) 对已经安装了的OpenSSH进行配置，首先我们开启 ssh 服务并设置开机启动以及设置相关的防火墙：

        7.  此时可以用用户和密码从另一台机器ssh本地

            虽然 PowerShell 比 Command 改进了不少，但是这里还是希望能够用上 bash。这里，我们就考虑使用 Git 附赠的 bash.exe 来代替 PowerShell.exe。找到你的 bash.exe 的位置，例如默认的配置是 "C:\Program Files\Git\bin\bash.exe"。 用这个地址去替换代码中的相应地址就好了，即：

            ```text
            New-ItemProperty -Path "HKLM:\SOFTWARE\OpenSSH" -Name DefaultShell -Value "D:\sk\software\cmder\Cmder.exe" -PropertyType String -Force
            ```

## VMware安装Linux

- 下载VMware
- 下载Ubuntu

![选择Ubuntu iso文件](.\Linux笔记记录.assets\VM_1.png)

![VM安装导向](.\Linux笔记记录.assets\create_new_VM.png)
![VM安装导向](.\Linux笔记记录.assets\create_new_VM2.png)

![VM安装导向](.\Linux笔记记录.assets\create_new_VM3.png)

![VM安装导向](.\Linux笔记记录.assets\create_new_VM4.png)

![finishInstall](.\Linux笔记记录.assets\finishInstallUbuntu'.png)

## Linux简单命令

`uname -a`: 查看linux架构

`ls -lht`: 列出当前目录下每个文件的大小, 同时也会给出当前目录下所有文件大小总和

`du -sh *`: 也会列出当前文件夹下所有文件对应的大小. [把*替换为具体的文件名, 会给出具体文件的大小] 

### screen

`screen`: 用于命令行终端切换;

````
apt-get install screen

#1. screen -S yourname: 新建一个叫yourname的session
#2. screen -ls: 列出所有的session
#3. screen -r yourname: 回到yourname这个session
#4. screen -d -r yournmae: 结束当前session, 并回到yourname这个session
#4. ctrl+A+C 重新启动一个session
#5. 所有的快捷键, 都是以ctrl+A 开始
````

-   **Ctrl + a c**：创建一个新窗口（带shell）

-   **Ctrl + a “**：列出所有窗口

-   **Ctrl + a 0**：切换到窗口0（按数字）

-   **Ctrl + a**：重命名当前窗口

-   **Ctrl + a S**：将当前区域水平分为两个区域

-   **Ctrl + a |** ：将当前区域垂直分为两个区域

-   **Ctrl + a**：选项卡将输入焦点切换到下一个区域

-   **Ctrl + a Ctrl + a**：在当前区域和上一个区域之间切换

-   **Ctrl + a Q**：关闭除当前区域外的所有区域

-   **Ctrl + a X**：关闭当前区域

-   **从Linux Screen 回话中脱离**

    您可以随时输入以下内容从Screen 会话中分离：

    ```javascript
    Ctrl+a d
    ```

    从Screen 会话中分离后，在Screen 会话中运行的程序将继续运行。

    **重新连接到Linux Screen**

    要恢复Screen会话，请使用以下命令：

    ```javascript
    screen -r
    ```

    如果您的计算机上正在运行多个Screen会话，则需要在**r参数**后附加屏幕会话ID。

    要查找会话ID，请使用以下命令列出当前正在运行的Screen会话：

    ```javascript
    screen -ls
    ```

    如果要还原屏幕16364.pts-0，请键入以下命令：

    ```javascript
    screen -r 16364
    ```

    复制

    ![img](Linux笔记记录.assets/w7u1fwxrm9.jpeg)

    **删除当前screen回话**

    进入当前回话，并输入：

    ```javascript
    exit
    ```

### mv

`mv /source/path/{.[!.],}* /destination/path`

花括号中的内容匹配, 并用都好分隔. `{a,b}*`会找到所有以a或b开头的文件. 

第二个匹配项只是一个空匹配项, 等效于`*`, 2⃣️第一个匹配项则等效于`.[!.]`,其中组`[!.]`表示组不是以开头的`.`

```bash
-f：强制覆盖，如果目标文件已经存在，则不询问，直接强制覆盖；
-i：交互移动，如果目标文件已经存在，则询问用户是否覆盖（默认选项）；
-n：如果目标文件已经存在，则不会覆盖移动，而且不询问用户；
-v：显示文件或目录的移动过程；
```



**ln -s**

ln是link的缩写, 在linux中ln命令的功能是为了某一个文件在另一位置建立一个同步的链接. 

目录只能创建软链接。
目录创建链接必须用绝对路径，相对路径创建会不成功，会提示"符号连接的层数过多"这样的错误。

```bash
-b或--backup   删除，覆盖目标文件之前的备份。 
-d或-F或--directory   建立目录的硬连接。 
-f或--force 　强行建立文件或目录的连接，不论文件或目录是否存在。 
-i或--interactive 　覆盖既有文件之前先询问用户。 
-n或--no-dereference 　把符号连接的目的目录视为一般文件。 
-s或--symbolic 　对源文件建立软链接(符号连接)，而非硬连接。 
-S<字尾备份字符串>或--suffix=<字尾备份字符串> 　用"-b"参数备份目标文件后，备份文件的字尾会被加上一个备份字符串，预设的字尾备份字符串是符号"~"，可通过"-S"参数来改变它。 
-v或--verbose 　显示指令执行过程。 
-V<备份方式>或--version-control=<备份方式> 　用"-b"参数备份目标文件后，备份文件的字尾会被加上一个备份字符串，这个字符串不仅可用"-S"参数变更，当使用"-V"参数<备份方式>指定不同备份方式时，也会产生不同字尾的备份字符串。 
```

**du**
用来查看目录或文件所占用磁盘空间的大小: `du -sh \path` 

　　**-h**：以人类可读的方式显示

　　**-a**：显示目录占用的磁盘空间大小，还要显示其下目录和文件占用磁盘空间的大小

　　**-s**：显示目录占用的磁盘空间大小，不要显示其下子目录和文件占用的磁盘空间大小

　　**-c**：显示几个目录或文件占用的磁盘空间大小，还要统计它们的总和

　　**--apparent-size：**显示目录或文件自身的大小

　　**-l** ：统计硬链接占用磁盘空间的大小

　　**-L**：统计符号链接所指向的文件占用的磁盘空间大小

**软链接多个文件-批量**
一个命令即可把一个文件夹下的所有文件软连接到另一个文件夹下.

```bash
ln -s source/* target/
# 例如将/jdfssz2/ST_BIOINTEL/P20Z10200N0039/06.groups/sukui目录下的所有文件软连接到/home/sukui/目录下
rm -rf ~/.*
ln -s /jdfssz2/ST_BIOINTEL/P20Z10200N0039/06.groups/sukui/* ~/
```

注意:

```bash
ln -s /jdfssz2/ST_BIOINTEL/P20Z10200N0039/06.groups/sukui/* ~/
rm -rf sukui/#虽然是想删除软连接, 但是由于进入到sukui目录下, 再进行删除, 所以会删除sukui/路径下的所有文件.
```

正确的删除方式(删除软链接, 但不删除实际数据)

```bash
rm -rf ./test_chk_ln
```

![img](Linux笔记记录.assets/Center.jpeg)

错误的删除方式

```bash
rm -rf ./test_chk_ln/ #这样就会把原来test_chk下的内容删除
```

![img](Linux笔记记录.assets/Center-16594284065843.jpeg)

注意: 再删除软链接过程中, 不需要`r`, 因为软链接只是文件, 不需要`r`, 带上`r`很可能把软链接指向的文件夹删除了. 所以`rm -f` 甚至`rm`更好, 还有删除内容的反馈.

**批量删除失效软链接**

```sh
find -type l -delet [PATH]
```

可删除[PATH]路径下的所有软连接.

type限定了文件类型为l(links), 加上-delete参数就可以实现找出所有软链接并删除

如果要删除失效的软链接, 需要加上一个`-L`参数, 表示追踪软链接.

```sh
find -L [PATH] -type l -delete
```



### 安装anaconda

-   下载anaconda安装包

    anaconda的[版本](https://repo.anaconda.com/archive), 最新版本为[anaconda](https://repo.anaconda.com/archive/Anaconda3-4.4.0-Linux-x86_64.sh), 下载命令

    ```bash
    wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
    ```

    ![image-20220511190605767](Linux笔记记录.assets/linux_anaconda_install.png)

-   解决安装出现的bug

    当显示`wget: command no found`, 则需要安装`wget`
    ```bash
    #centos
    apt-get install -y wget
    #centos
    yum install wget -y
    ```

-   安装anaconda

    需要首先赋权再执行安装程序: 

    ```bash
    chmod+ +x Anaconda3-2022.5-Linux-x86_64.sh
    ./Anaconda3-2022.5-Linux-x86_64.sh
    #1. 持续回车同意
    #2. 输入yes, 添加到环境变量
    ```

    若跳过环境变量设置, 则需要自己到文件夹设置Anaconda安装路径: `/home/xxx/.bashrc`, 并再最后一行添加`export PATH=/home/xx/anaconda3/bin:$PATH`

-   完成安装并刷新

    ```bash
    source ~/.bashrc
    conda --version
    conda create -n py37 python=3.7
    conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
    
    ```

### **SCP**

-   从服务器上下载文件

    `scp username@servername:/path/filename /var/www/local_dir`

-   上传本地文件到服务器

    `scp /path/filename username@servername:/path  `

-   从服务器下载整个目录

    `scp -r username@servername:/var/www/remote_dir/（远程目录） /var/www/local_dir（本地目录）`

-   4、上传目录到服务器
    `scp -r local_dir username@servername:remote_dir`
    
-   将本地文件复制多个文件到远程主机(多个文件使用空格分隔开)

    `scp /localfold1/index.css /localfold2/json.js sukui@192.168.61.6:~/1_data`

-   从远程主机复制多个文件到当前文件目录

    `scp sukui@192.168.61.6:/usr/local/data/\{index.css,json.js\}`#不能有空格

    `scp remoteuser@remotehost:"/remotefold1/file1 /remotefold2/file2" /localfold/ `

    注: 本地路径或者远程路径不能使用`~`符号.


### vi

| 命令            | description                            |
| --------------- | -------------------------------------- |
| vi filename     | 打开或新建文件, 并将光标置于第一行     |
| vi + n filename | 打开文件, 并将光标置于第n行            |
| h               | 光标左移一个字符                       |
| I               | 光标右移一个字符                       |
| k/ ctrl+p       | 光标上移                               |
| j/ Ctrl+n       | 光标下移                               |
| i               | 在光标前插入文本                       |
| I               | 在当前行首插入文本                     |
| do              | 删除至首行                             |
| x or X          | 删除一个字符,x删除光标后,而X删除光标前 |
| ctrl+u          | 删除输入方式所输入的文本               |
| : w             | 保存当前文件                           |
| :e filename     | 打开文件filename进行编辑               |
| :x              | 保存当前文件并退出                     |
| :q              | 退出vi                                 |
| :q!             | 不保存退出vi                           |
|                 |                                        |

### sudo

Ubuntu系统有一个与众不同的特点,初次使用时, 无法作为root来登录系统. 一般是通过`sudo`命令授予普通用户root权限.

- sudo缺省配置

  默认时, Ubuntu为sudo提供了一个基本的配置, 该配置保存在`/etc/sudoers`文件中

  ```
  #User privilege specification
  root ALL = (ALL)ALL
  testuser ALL = (root)/usr/bin/find,/bin/rm
  testuser ALL = (root)NOPASSWD:/bin/kill,/usr/bin/killall
  
  #Members of the admin group may gain root privileges
  %admin ALL = (ALL)ALL
  %accounts ALL=(root)/usr/sbin/useradd,/usr/sbin/userdel./usr/sbin/usermod
  ```

  第一项配置作用: 是允许root用户使用sudo命令变成系统中任何其他类型的用户

  第二项配置规定: 管理组中所有成员能以root的身份只需所有命令

  因此, 在默认安装的Ubuntu系统中, 要想作为root身份来只需命令的话, 只要在sudo后面更上欲执行的命令即可.

- 

### 查看当前目录文件大小

`du=disk usage`, 磁盘使用率, 输出每个文件或目录总大小

```bash
du [option] [FILE]
#-c: total累加大小
#-d: --max_depth=N, 决定统计每个目录的深度
#-h: --human-readable以高刻度方式打印
#-s: --summarize显示总大小
du -sh#查看当前目录总大小
du -sh * #查看当前目录所有子目录大小
du -ach * #查看当前目录和所有子目录大小, 最后一行会显示当前目录的总大小, 不包括隐藏文件
```

**df**
`df=disk free`, 用于显示磁盘分区上的可使用的磁盘空间. 该命令来获取硬盘被占用了多少空间

`df -h`: 以高可读的结果展示磁盘分区上的可使用的磁盘空间

文件系统 容量 已用 可用 已用占比 挂载点

![df -h](Linux笔记记录.assets/05664db9bf674a1f8976dbe76bc994c3.png)



### 配置环境

- 修改解析域名

  `vi /etc/systemd/resolved.conf`

  ![img](.\Linux笔记记录.assets\editConf.png)

  保存后, 重启

- 更新源: `sudo apt-get update; sudo apt-get upgrade`

- 配置清华源: 

  ```
  首先把ubuntu的软件源改成163 ， 操作如下：
  cd ..进入到根目录下（可能需要多次执行）
  cd /etc   #进入etc配置文件目录
  cd apt  #进入apt目录
  ls       #查看所有文件目录
  cat sources.list  #查看文件内容（需要将里边的内容替换成ubuntu清华源:https://mirrors.tuna.tsinghua.edu.cn/help/ubuntu/）
  sudo cp sources.list sources.list.bak  #养成良好习惯，在替换之前，备份sources.list文件
  sudo echo '' > sources.list  #可能会发现sudo命令也用不了，因为这个文件在root用户root组下
  ls -al | grep sources.list  
  su root //那么，切换到root目录下操作
  sudo passwd root //如果su命令不能切换root，提示su: Authentication failure，只要你sudo passwd root过
                    一次之后，下次再su的时候只要输入密码就可以成功登录了。
  sudo echo '' > sources.list  #重新删除内容（用空内容覆盖）
  cat sources.list  #再次查看文件内容，已经为空
  nano sources.list  #进入文件
         alt + ctrl键盘退出，复制https://mirrors.tuna.tsinghua.edu.cn/help/ubuntu/搜索到的内容
         注意不能用ctrl+v粘贴，得从选项卡操作：编辑-粘贴，
         然后删除空行
  ctrl+o/ctrl+s保存 回车
  ctrl+x 退出
  cat sources.list //再次查看文件内容，应该已经替换成163的内容了
  su ubuntu #切回到ubuntu目录
  sudo apt-get update //软件源更新
  ```

- 安装vim: `sudo apt-get install vim`

- 安装net-tool: `sudo apt-get install net-tools`

- 安装ssh服务: `sudo apt-get install openssh-server`

## VI

![img](Linux笔记记录.assets/20210806102207186.gif)

**VIM命令**

```bash
:set nu#显示行号
:set nonu#取消行号
:n #定位到n行
```

## 目录与路径

**相关目录操作**

```bash
. #代表此层目录
.. #上一层目录
- #前一个工作目录
~ #目前用户身份在的家目录
```

几个常见的处理目录的指令

```bash
cd #变换目录
pwd #显示当前目录
mkdir #建立一个新的目录
rmdir #删除一个空目录
```

**关于执行文件路径的变量**
`ls`指令完整文件名为`/bin/ls`, 其可以在任意地方执行是因为换了变量`PATH`的能力.

当我们执行一个指令的时候, 系统会将`PATH`的设定去每个PATH定义的目录下搜索对应的可执行文件, 如果在PATH定义的目录中含有多个名为ls的可执行文件, 那么先搜索到的被执行.

```bash
echo $PATH
/opt/openmpi/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/opt/ganglia/bin:/opt/ganglia/sbin:/usr/java/latest/bin:/opt/rocks/bin:/opt/rocks/sbin:/opt/condor/bin:/opt/condor/sbin:/opt/gridengine/bin/lx-amd64:/home/sukui/.local/bin:/home/sukui/bin
```

**cp, rm, mv**
选项参数

```bash
a #相当于-dr
d #若源文件为链接文件的属性(link file), 则复制链接文件属性而非文件本身
f #强制(force), 若目标文件已经存在且无法开启, 则移除后再尝试依次
i #若目标文件已经存在时, 在覆盖时会先询问动作的进行.
I #进行硬式链接的复制, 而非复制文件本身
p #连通文件的属性(权限, 用户, 时间)一起复制, 而非使用默认属性
r #递归持续复制, 用于目录的复制行为
s #复制称为符号链接文件(symbolic link)
```

创建符号链接与实体链接

```bash
# 将之前复制过来的 bashrc 建立一个链接档
[root@study tmp]# pwd
/tmp
# 先查看该文件的属性
[root@study tmp]# ls -l bashrc
-rw-r--r--. 1 root root 176 10月 13 15:38 bashrc
# 分别建立 符号链接 和硬式链接
[root@study tmp]# cp -s bashrc bashrc_link
[root@study tmp]# cp -l bashrc bashrc_hlink
[root@study tmp]# ls -l bashrc*
# 注意看这里的数值，源文件是 1 这里变成了 2
-rw-r--r--. 2 root root 176 10月 13 15:38 bashrc
-rw-r--r--. 2 root root 176 10月 13 15:38 bashrc_hlink
# 下面这条数据，在终端中，bashrc_link 会显示浅蓝色
lrwxrwxrwx. 1 root root   6 10月 13 15:55 bashrc_link -> bashrc
```

使用`-l`及`-s`都会建立连接档(link file), 那么有什么不同

1.   bashrc_link: 使用s创建出来, 是符号链接(symbolic link), 简单说是一个快捷方式, 会链接到bashrc中, 有一个`->`的符号
2.   bashrc_hlink: 使用l创建出来, 是实体链接(hard link)

备份复制: 当源文件比目标新的时候才复制

```sh
# 先查看两个文件的时间，可以看到源文件是 2013 年，比目标文件旧
[root@study tmp]# ls -l ~/.bashrc /tmp/bashrc
-rw-r--r--. 1 root root 176 12月 29 2013 /root/.bashrc
-rw-r--r--. 2 root root 176 10月 13 15:38 /tmp/bashrc
# 这里使用 -u 复制后，没有任何提示
[root@study tmp]# cp -u ~/.bashrc /tmp/bashrc
# 再次查看，发现没有复制成功，当前时间是 16:14 了，如果成功，目标文件的时间也会变更
[root@study tmp]# ls -l ~/.bashrc /tmp/bashrc
-rw-r--r--. 1 root root 176 12月 29 2013 /root/.bashrc
-rw-r--r--. 2 root root 176 10月 13 15:38 /tmp/bashrc
```

链接文档复制

```bash
# 该文件是一个符号链接文件
[root@study tmp]# ls -l bashrc_link
lrwxrwxrwx. 1 root root 6 10月 13 15:55 bashrc_link -> bashrc
# 这里使用不加参数复制和加参数复制
[root@study tmp]# cp bashrc_link bashrc_link_1
[root@study tmp]# cp -d bashrc_link bashrc_link_2
[root@study tmp]# ls -l bashrc bashrc_link*
-rw-r--r--. 2 root root 176 10月 13 15:38 bashrc
lrwxrwxrwx. 1 root root   6 10月 13 15:55 bashrc_link -> bashrc
# 可以看到，不加参数复制把源文件复制过来了
-rw-r--r--. 1 root root 176 10月 13 16:16 bashrc_link_1
# 添加 -d 参数，只复制了链接文件本身
lrwxrwxrwx. 1 root root   6 10月 13 16:16 bashrc_link_2 -> bashrc
```

rm删除文件或目录

```bash
rm [-fir] 文件或目录
#f: 强制删除, 忽略不存在的文件
#i: 互动模式, 在删除时会询问使用者是否操作
#r: 递归删除
```

mv移动文件与目录或更名

```bash
mv [-fiu] source destination
mv [options] source1 source2 ... directory
#u: 若目标已存在, 且source比较新, 才会成功更新该文件
```

取得路径的文件名与目录名称: `basename, dirname`
```bash
# /etc/sysconfig/network 比如这个路径
# 可以使用指令分别获取到他的目录与文件名
[root@study tmp]# basename /etc/sysconfig/network
network
[root@study tmp]# dirname /etc/sysconfig/network
/etc/sysconfig
```

### 文件内容查阅

-   `cat`: 由第一行开始显示文件内容
-   `tac`: 从最后一行开始显示
-   `nl`: 显示时附带行号
-   ‘head’: 只看头几行

```bash
cat [-AbEnTv]
#A: 相当于-vET的整合, 可以列出一些特殊字符而不是空白
#b: 理出行号, 针对非空白行做行号显示
#E: 将结尾的断行字符$显示出来
```

### 文件与目录的默认权限与隐藏权限

基本权限(rwx), 是否为目录(d), 文件(-)或者链接(l)等属性, 修改属性可通过`chgrp, chown, chmod`.

`umask`: 指定目前用户在建立文件或目录时候的默认权限

```bash
# 以数值形态显示
[mrcode@study tmp]$ umask
0002   # 与一般权限有关的是后面三个数字

# 还可以以符号来显示
[mrcode@study tmp]$ umask -S
u=rwx,g=rwx,o=rx
```

在数值形态有四组, 第一组时特殊权限. 

1.   文件: 没有可执行(x)权限, 只有rw, 也即是最大为666`-rw-rw-rw`
2.   目录: 由于x与是否可以进入此目录有关, 隐藏默认所有的权限均开发, 即777: `xrwxrwxrw`

```markdown
r,w,x分别是4,2,1分
002, 也就是oters的权限被拿掉了2, 也就是w, 权限如下
建立文件时: 预设 -rw-rw-rw, 减掉2变成了-rw-rw-r--
```

**umask的利用与重要性**
如果umask设置为0022, 那么相当于group默认创建只有r属性, 除了拥有者, 其他人只能读, 不能写. 所以需要修改umask的值

```bash
# 修改语法是 umask 后接数值
# 由于笔者的 centos 较新，默认已经是 002 的了，这里就更改回 022 来测试
[mrcode@study tmp]$ umask
0002
[mrcode@study tmp]$ umask 022   # 更改为 022
[mrcode@study tmp]$ umask
0022
[mrcode@study tmp]$ touch test3
[mrcode@study tmp]$ mkdir test4
[mrcode@study tmp]$ ll -d test[34]  # 使用正则来匹配 test3和4
# 可以看到 文件 group 和 ohters 已经没有了 w
-rw-r--r--. 1 mrcode mrcode 0 Oct 13 22:23 test3
drwxr-xr-x. 2 mrcode mrcode 6 Oct 13 22:23 test4
```

### 脚本文件名的搜索

**which搜索执行文档**

```markdown
which [-a] command
-a: 将所有PATH目录中可以找到的指令均输出, 而不止第一个被找到的指令名称
```

```bash
# 搜索 ifconfig 这个指令完整文件名
[root@study tmp]# which ifconfig
/sbin/ifconfig

# 查看 which 在哪个位置
[root@study tmp]# which which
alias which='alias | /usr/bin/which --tty-only --read-alias --show-dot --show-tilde'
        /bin/alias
        /usr/bin/which

# 这里发现了两个 which，其中一个是 alias
# alias 是指令的别名，输入 which 后，就等于属于了等于后面的那一串指令
# 更多的数据在后续的 bash 章节中讲解

# 找出 history 指令的完整文档名
[root@study tmp]# which history
/usr/bin/which: no history in (/usr/lib64/qt-3.3/bin:/usr/local/sbin:/usr/local/bin:/sbin:/bin:/usr/sbin:/usr/bin:/root/bin)
# 上面在列出的目录中没有找到
# 下面报错没有 --help 这个选项
[root@study tmp]# history --help
-bash: history: --: 无效选项
history: 用法:history [-c] [-d 偏移量] [n] 或 history -anrw [文件名] 或 history -ps 参数 [参数...]
```

**文件名的搜索**
linux中许多搜索指令, 通常find不很常用, 因为速度慢. 所以先用`whereis`或者是`locate`, 如果找不到, 再用`find`

```markdown
whereis: 只找系统中某些特定目录下的文件, 速度快
locate: 利用数据库来搜索文件名
find 搜索全盘内的文件状态
```

```bash
whereis [-bmsu] 文件名或目录
#l: 列出whereis回去查询的个几个主要目录
#b: 只找binary格式的文件
#m: 只找说明文件manual路径下的文件
#s: 只找source来源文件
```

```bash
# 找到 ifconfig 文件名
# 下面发现找到了两个包含 ifconfig 的
[root@study tmp]# whereis ifconfig
ifconfig: /usr/sbin/ifconfig /usr/share/man/man8/ifconfig.8.gz

# 找到与 passwd 有关的说明文件文件名（man page）
[root@study tmp]# whereis passwd
passwd: /usr/bin/passwd /etc/passwd /usr/share/man/man1/passwd.1.gz /usr/share/man/man5/passwd.5.gz

# 这里添加 -m 参数就只找说明文件了
[root@study tmp]# whereis -m passwd
passwd: /usr/share/man/man1/passwd.1.gz /usr/share/man/man5/passwd.5.gz
```

`whereis`主要是针对`/bin/sbin`下的执行文件, `/usr/share/man`下的man page文件, 和几个特定的目录, 所以速度很快. 可以使用whereis -l来显示具体会找哪些目录

**locate / updatedb**

```bash
locate [-ir] keyword
#i: 忽略大小写的差异
#c: 不输出文件名, 仅计算找到的文件数量
#l: 仅输出几行, 如-l 5表输出5行
#S: 输出locate所使用的数据库文件相关信息, 包括该数据库记录的文件/目录数量
```

```bash
# 找出系统中所有与 passwd 先关的文件名，且只列出 5 个
[root@study tmp]# locate -l 5 passwd
/etc/passwd
/etc/passwd-
/etc/pam.d/passwd
/etc/security/opasswd
/usr/bin/gpasswd
# 可以看到找到了 9 前多个相关的
[root@study tmp]# locate -c 5 passwd
9863

# 列出 locate 查询所使用的数据库文件与各数据数量
[root@study tmp]# locate -S
数据库 /var/lib/mlocate/mlocate.db:
        11,623 文件夹
        153,170 文件
        7,466,565 文件名中的字节数
        3,468,856 字节用于存储数据库
```

locate其实是模糊搜索, 只要包含关键字的文件名都会被匹配, 它有限制, 查找的数据库是已建立的数据库`/var/lib/mlocate`里的数据来搜索

该数据库建立默认每天搜索一次. 可以手动触发数据库更新.`updatedb`

**find**
`find [path] [option] [action]`

与时间有关的参数`-atime, -ctime, -mtime`

```bash
# 将过去系统上 24 小时内有更动过内容（mtime）的文件列出
find / -mtime 0
# 0 表示当前时间，也就是当前时间开始往前 24 小时，也就是 24 小时内被修改过的文件

# 3 天前，24 小时内，如下
find / -mtime 3

# 寻找 /etc 下的文件，如果文件日期比 /etc/passwd 新旧列出
find /etc -newer /etc/passwd

# 列出 4 天内被更动多的文件
find / -mtime -4
```

![img](Linux笔记记录.assets/markdown-img-paste-20191017222127697.261ce271.png)

mtime选项的n正负数表示不同的含义

### BASH

变量再程序中非常重要, 如在shell script中, 前面几行定义变量, 后面的大量逻辑处理使用变量.

**变量的取用与设置: echo, 变量设置规则, unset**

```bash
echo $variable
echo $PATH
echo ${PATH} #推荐使用
```

变量的修改使用等号赋值

```bash
[mrcode@study ~]$ echo ${myname}
											# 这里没有任何数据，该变量不存在或未设置
[mrcode@study ~]$ myname=mrcode
[mrcode@study ~]$ echo ${myname}
mrcode

# 在 bash 中，当一个变量名称未被设置时，预设内容就是 空
```

变量内容若有空格可以使用双引号或单引号限定, 但是一下除外

1.   [$] 在双引号中可以保留原本的特性

     ```bash
     var="lang is $LANG"
     则使用 echo $var 则得到输出信息为：lang is utf8 等的字样
     ```

2.   [‘]在单引号内的特殊字符仅为一般字符

     ```bash
     var='lang is $LANG'
     则输出信息为：lang is $LANG
     ```

3.   [\\] 跳脱字符

     ```bash
     # 就是转义符
     myname=mrcode\ Tsai  # 这里将空格转义成普通字符了
     ```

4.   在一串指令的执行中, 还需要使用其他额外的指令所提供的信息时, 可以使用反单引号

     ```bash
     # 取得核心版本的设置
     [mrcode@study ~]$ echo  $version
     
     [mrcode@study ~]$ version=$(uname -r); echo $version
     3.10.0-1062.el7.x86_64
     ```

5.   若该变量为扩展变量内容时, 可以使用如下方式累加变量

     ```bash
     PATH="$PATH:/home/bin"
     PATH=${PATH}:/home/bin
     ```

6.   取消变量使用`unset`
     ```bash
     #取消myname的设置
     unset myname
     ```

7.   若改变量需要再其他子程序中执行, 则需要使用export来使变量变成环境变量

     ```bash
     export PATH
     ```

### 命名别名与历史命令

**alias, unalias**
如`ls -al | more` 查看隐藏文档并且翻页查看, 觉得这个命令太长了, 可以通过设置别名

```bash
[mrcode@study ~]$ alias lm='ls -al | more'
[mrcode@study ~]$ lm
total 68
drwx------. 18 mrcode mrcode  4096 Nov 11 10:20 .
drwxr-xr-x.  4 root   root      42 Oct  8 23:01 ..
-rw-------.  1 mrcode mrcode 10279 Nov 11 14:12 .bash_history
-rw-r--r--.  1 mrcode mrcode    18 Aug  8 20:06 .bash_logout
-rw-r--r--.  1 mrcode mrcode   193 Aug  8 20:06 .bash_profile
```

别名的定义规则与变量定义规则几乎相同, 另外可以取代已经存在的变量名

`alias rm='rm -i'`

### 路径与指令搜索顺序

1.   以相对路径, 绝对路径执行命令, 例如`/bin/ls`或`./ls`
2.   由alias找到该指令来执行
3.   由bash内置的指令执行
4.   通过`$PATH`这个变量的顺序搜索到的第一个指令执行

也可使用`type -a ls`来查询指令搜寻的顺序

```bash
# 范例：设置 echo 的命令别名为 echo -n，然后观察 echo 执行的顺序
[mrcode@study ~]$ alias echo='echo -n'
[mrcode@study ~]$ type -a echo
echo is aliased to `echo -n'
echo is a shell builtin
echo is /usr/bin/echo
```

**source: 读取环境配置文件的指令**
由于`/etc/profile`与`~/.bash_profile`都是在取得login shell的时候才会读取的配置文件, 所以将自己的偏好设置写入上述文件后, 通常都是需要注销后才会生效. 可以使用source指令达到立即生效.

```bash
# 范例：将 家目录的 ~/.bashrc 的设置读入目前的 bash 环境中
[mrcode@study ~]$ source ~/.bashrc 
[mrcode@study ~]$ . ~/.bashrc 
# 使用 source 或则 小数点的语法 都能将内容读取到当前的 shell 环境中
```

### 管线命令

假设想知道`/etc`下有多少文件, 可以使用`ls /etc/`来查询, 不过因为文件太多, 输出占满整个屏幕, 我们可以通过管线命令结合less

```bash
ls -al | less
```

如此一来, `ls -al`指令输出后的内容, 能够被less读取, 并且利用less的功能, 可以前后反动相关信息.

管线命令仅能处理由前一个指令传来的正确信息, 对于standard error没有直接的处理能力.

![image-20191204232521041](Linux笔记记录.assets/image-20191204232521041.fd19ab13.png)

在每个管线后面接的第一个数据必定是指令, 而且整个指令必须能够接收standard input的数据才可以, 这样的指令称为[管线命令]. 例如less, more, head, tail等都是可以接受standard input的管线命令. 而ls, cp, mv等就不是管线命令.

**grep**
cut是将一行信息中, 取出某部分想要的数据, 而grep则是分析一堆信息, 若一行当中由匹配的数据, 则将这一行数据拿出来.

```bash
grep [-acinv] [--color=auto] "搜索字符串" filename
#a: 将binary文件以text文件的方式搜索
#c: 计算找到[搜索字符]的次数
#i: 忽略大小写
#n: 输出行号
#v: 反向旋转, 显示没有搜索字符串的那一行数据
```

```bash
# 范例 1：将 last 中，有出现 root 的那一行找出来
[mrcode@study ~]$ last | grep 'root'
root     tty3                          Sun Oct  6 23:16 - crash  (22:40)    
root     tty4                          Fri Oct  4 22:48 - 22:48  (00:00)  
# 会发现 root 被高亮颜色了，我们时候 type 命令查看，发现被自动加上了 color 参数
[mrcode@study ~]$ type grep
grep is aliased to 'grep --color=auto'

# 范例 2：与 范例 1 相反，不要 root 的数据
[mrcode@study ~]$ last | grep -v 'root'
mrcode   pts/1        192.168.0.105    Mon Dec  2 01:25   still logged in   
mrcode   pts/0        192.168.0.105    Mon Dec  2 01:25   still logged in   
mrcode   pts/1        192.168.0.105    Mon Dec  2 00:21 - 01:12  (00:51)
reboot   system boot  3.10.0-1062.el7. Fri Oct  4 18:47 - 03:43  (08:56)

# 范例 3：在 last 的输出信息中，只要有 root 就取出，并且只取第一栏
# 结合 cut 命令取出第一栏
[mrcode@study ~]$ last | grep 'root' | cut -d ' ' -f 1
root
root

# 范例 4：取出 /etc/man_db.conf 内涵 MANPATH 的那几行
[mrcode@study ~]$ grep 'MANPATH' /etc/man_db.conf
# MANDATORY_MANPATH                     manpath_element
# MANPATH_MAP           path_element    manpath_element
```



## 安装Zsh

```ssh
#安装
sudo apt-get install zsh

#将zsh设置为默认shell
chsh -s /bin/zsh

#检查
echo $SHELL
#返回/usr/bin/zsh 即表示成功; 若没成功,重启试试
```

- 安装Oh My Zsh

  Oh My Zsh时基于Zsh命令行的一个扩展工具集, 提供了丰富的扩展功能, 如: 主题配置, 插件机制, 内置的便捷操作.

  ```python
  # 通过 curl
  sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
  
  # 或者 通过 wget
  sh -c "$(wget -O- https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
  
  ```

  ![image-20211216173001374](.\Linux笔记记录.assets\zsh.png)

### Python环境配置

安装python: `sudo apt-get install python3.7`

查看版本号: `python3.7 --version`

查找python: `which python3.7`#系统搜索命令时, 按照PATH环境变量顺序依次进行搜索, `/usr/local/bin`,`/usr/local/bin/python`

在Ubuntu环境中, 可以直接使用apt进行pip安装, 或者可以通过easy_install进行安装, 需要注意. Python2 对应pip, python 3 对应的是pip3

```python
sudo apt-get install python3-pip python3-dev build-essential
pip3 install --upgrade pip3
```

- 配置源

  在国内环境如果直接使用官方的pip源安装软件,非常慢, 可以转为清华源

  pip配置文件为`~/.pip/pip.conf`,默认是不存在的,需要手动创建.

  ```
  [global]
  trusted-host = mirrors.aliyun.com
  index-url = http://mirrors.aliyun.com/pypi/simple
  ```

- 配置代理

  内外环境下安装软件包时,出现"cannot fetch index base URL"

  ```
  pip --proxy=http://username:passed@proxy_host install package
  ```

- 安装pytorch

  ```
  pip3 install --upgrade pip
  pip3 --version
  pip3 --no-cache-dir install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
  ```

  #`--no-cache-dir`: 解决killed问题, 必须保证分配的虚拟机内存>3G

#### python删除

1. 卸载python3.4 : `sudo apt-get remove python3.6`
2. 卸载python3.4及其依赖: `sudo apt-get remove --auto-remove python3.6`
3. 清除python3.4: `sudo apt-get purge python3.6`

## Linux安装python



## Linux安装Docker

基本概念

- 从Docker下载下来的叫做镜像(images)
- 使用`docker run` 运行起来的镜像(images)叫容器(containers)

### 第一步-安装Docker

官方Ubuntu存储库中提供的Docker安装软件包可能不是最新版

Ubuntu官方的版本库中并不是Docker最新包,为了保证最新版本, 我们从Docker官方库来安装

1. 更新现有的软件包列表: `sudo apt update`#注: 如果无法更新, 可能时软件源指向的是国外服务器, 很可能被墙. 

2. 安装必备软件包, 让apt通过https使用软件包: `sudo apt-get install apt-transport-https ca-certificates curl software-properties-common`

3. 将官方Docker版本库的GPG密钥添加到系统中: `curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -`

4. 将Docker版本库添加到APT源:`sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu focal stable"`

   执行后显示

   ```python
   Get:1 https://download.docker.com/linux/ubuntu focal InRelease [36.2 kB]                                
   Hit:2 http://cn.archive.ubuntu.com/ubuntu focal InRelease                                               
   Get:3 https://download.docker.com/linux/ubuntu focal/stable amd64 Packages [3056 B]
   Hit:4 http://cn.archive.ubuntu.com/ubuntu focal-updates InRelease
   Hit:5 http://cn.archive.ubuntu.com/ubuntu focal-backports InRelease
   Hit:6 http://cn.archive.ubuntu.com/ubuntu focal-security InRelease
   Fetched 39.2 kB in 2s (16.1 kB/s)
   Reading package lists... Done
   ```

5. 用新添加的Docker软件包来进行升级更新: `sudo apt update`

6. 确保要从Docker版本库, 而不是默认的Ubuntu版本进行安装: `apt-cache policy docker-ce`

   执行后结果

   ```python
   docker-ce:
     Installed: (none)
     Candidate: 5:19.03.12~3-0~ubuntu-focal
     Version table:
        5:19.03.12~3-0~ubuntu-focal 500
           500 https://download.docker.com/linux/ubuntu focal/stable amd64 Packages
        5:19.03.11~3-0~ubuntu-focal 500
           500 https://download.docker.com/linux/ubuntu focal/stable amd64 Packages
        5:19.03.10~3-0~ubuntu-focal 500
           500 https://download.docker.com/linux/ubuntu focal/stable amd64 Packages
        5:19.03.9~3-0~ubuntu-focal 500
           500 https://download.docker.com/linux/ubuntu focal/stable amd64 Packages
   ```

   注意: 到这一步`docker-ce`还未安装, 但根据上一步中的列表, 可以看到docker-ce来自Dcoker官方版本库

7. 安装docker: `sudo apt install docker-ce`

8. Docker安装完毕, 然后启动守护程序, 检测Docker是否正在运行: `sudo systemctl status docker`

   ```python
   ● docker.service - Docker Application Container Engine
        Loaded: loaded (/lib/systemd/system/docker.service; enabled; vendor preset: enabled)
        Active: active (running) since Sat 2020-07-08 02:55:20 UTC; 5min ago
   TriggeredBy: ● docker.socket
          Docs: https://docs.docker.com
      Main PID: 4287 (dockerd)
         Tasks: 8
        Memory: 36.4M
        CGroup: /system.slice/docker.service
                └─4287 /usr/bin/dockerd -H fd:// --containerd=/run/containerd/containerd.sock
   ```

   现在, 安装Docker不仅可以为你提供Docker服务(守护程序), 还可以提供`docker`命令行使用程序.

### 如何在Docker中使用镜像

Docker(containers)是从Docker镜像中生成出来的, 默认情况下, Docker从[Docker Hub](https://hub.docker.com/)下周这些镜像, Docker公司在运营这个[Docker Hub](https://hub.docker.com/)

任何人都可以在Docker Hub上托管自己的Docker镜像. 因此, 我们大多数需要的应用程序和Linux发行版本都能在这里找到

需要检测是否可以访问Docker Hub和从这个网站下载镜像, 需输入`sudo docker run hello-world`

```python
Unable to find image 'hello-world:latest' locally
latest: Pulling from library/hello-world
2db29710123e: Pull complete 
Digest: sha256:cc15c5b292d8525effc0f89cb299f1804f3a725c8d05e158653a563f15e4f685
Status: Downloaded newer image for hello-world:latest

Hello from Docker!
This message shows that your installation appears to be working correctly.

To generate this message, Docker took the following steps:
 1. The Docker client contacted the Docker daemon.
 2. The Docker daemon pulled the "hello-world" image from the Docker Hub.
    (amd64)
 3. The Docker daemon created a new container from that image which runs the
    executable that produces the output you are currently reading.
 4. The Docker daemon streamed that output to the Docker client, which sent it
    to your terminal.

To try something more ambitious, you can run an Ubuntu container with:
 $ docker run -it ubuntu bash

Share images, automate workflows, and more with a free Docker ID:
 https://hub.docker.com/

For more examples and ideas, visit:
 https://docs.docker.com/get-started/

```

执行此命令时, Docker首先在本地查找`hello-world`, 如果没有, 它会从Docker Hub(默认版本库)下载该镜像. 下载镜像后, Docker会根据镜像创建一个容器, 并执行该容器中的应用程序

可通过将`docker`命令与`search`子命令配合使用来搜索Docker Hub上可用的镜像

例如, 要搜索Ubuntu的镜像, 请输入: `docker search ubuntu`

### 管理Docker容器

使用Docker一段时间后, 我们的计算机会有许多活动和非活动容器

要查看这些活动中的容器对象: `docker ps`

查看所有容器: `docker ps -a`

要查看最后创建的容器, 可使用:`docker ps -l`

```python
CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS                            PORTS               NAMES
7896ef8f403f        ubuntu              "/bin/bash"         20 minutes ago      Exited (127) About a minute ago                       keen_curie
```

要启动已停止的容器, 应使用`docker start+容器ID或容器名`: `docker start 7896ef8f403f`

要停止正在运行的容器,使用: `sudo docker stop 容器ID`: `sudo docker stop keen_curie`

若某容器不再使用, 我们可以通过容器ID或者容器名称来删除: `sudo docker rm keen_curie`

可以启动一个新容器, 并使用`--name`为其命名. 还可以使用`--rm`创建一个容器并让这个容器在结束时自动删除

### 卸载docker

在卸载Docker之前, 需移除所有的容器, 镜像, 卷和网络

运行下面的命令, 停止所有正在运行的容器, 并且移除所有docker对象

```
sudo docker container stop $(docker container ls -aq)
sudo docker system prune -a --volumes
```

现在可以使用`apt`像卸载其他软件包一样来卸载Docker

```
sudo apt purge docker-ce
sudo apt autoremove
```

## 将本地Python项目打包成docker镜像

1. 本地创建Python项目, 项目中创建hello.py

   ```python
   import numpy as np
   
   print(np.arange(1,10))
   print("Hello Docker~")
   #-------output-------------
   [1 2 3 4 5 6 7 8 9]
   Hello Docker~
   ```

2. 将项目依赖库进行导出, 使用`pipreqs ./ --encoding=utf-8 --force`

   ![image-20211216152833048](.\Linux笔记记录.assets\pipreqs.png)

3. 创建一个docker文件夹

   ![image-20211216153000057](.\Linux笔记记录.assets\dockfile.png)

4. app中的内容其实就是Python项目根据目录以及子目录文件, 咱们为了省事, 直接将先将新建的hello.py放到app目录下

5. 编写dockerfile, 创建一个文件叫做Dockerfile,内容如下

   ```dockerfile
   #基于的基础镜像
   FROM python:3.7.4
   #设置code文件夹是工作目录
   WORKDIR ./docker-test
   #代码添加到docker_demo文件夹
   ADD . .
   #安装支持
   RUN pip install -r requirements.txt
   CMD ["python","./app/hello.py"]
   ```

   - FROM python:3.7.4

     `FROM <基础镜像>`: 所谓定制镜像, 那一定是以一个镜像为基础. FROM指令用来指定以哪个镜像作为基础镜像生成新的镜像.

   - WORKDIR ./docker-test

     `WORKDIR <工作目录路径>`: 使用WORKDIR指令可以来指定镜像中的工作目录(或者称为docker的当前目录), 以后各层的当前目录就被改为指定的目录

   - ADD . .

     `ADD <源路径> <目标路径>`: 使用ADD指令可以将构建上下文目录中的源路径目录赋值到镜像内的<目标路径>位置. 

     第一个参数`.`: 代表Dockerfile所在的目录, 即python项目docker-test下所有的目录(不包括docker-test)

     第二个参数`.`: 代表镜像的工作目录docker-test.

     所以该行命令会将python项目docker-test目录下的所有文件复制到镜像的docker-test目录下. 这样docker镜像中就拥有一份docker-test python项目

   - RUN pip install -r requirements.txt

     RUN指令是用来执行命令行命令的, 这里直接按照requirements.txt中指定的任何所有软件包

   - CMD ["python","./app/hello.py"]

     CMD指令是容器启动命令, 这里是在容器启动时通过python运行hello.py. 值得注意的时`./`目录指的时当前docker工作目录`docker-test`

     

6. docker环境提前准备好

7. 在服务器中创建一个目录, docker-test, 将桌面上创建的docker-test中的内容, 传到服务器上

   ![image-20211216155018921](.\Linux笔记记录.assets\docker-test-file.png)

   ![image-20211216155632640](.\Linux笔记记录.assets\docker-test-file2.png)

8. 执行`docker build -t hello-test .`, 这里注意, hello-test后面有一个`.`

   利用`build`指令生成镜像,-t参数指定了最终镜像名称为hell-test. `.`表示当前目录, 这里是指将Dockerfile所在的当前目录作为构建上下文目录.

   ![image-20211216161858027](.\Linux笔记记录.assets\run-docker.png)

9. 执行`docker run -it hello-test`, 可以看到输出python的执行结果

   用`docker run`运行容器, 终端显示hello Docker~表示容器运行成功

   ![image-20211216164240090](.\Linux笔记记录.assets\docker-result.png)

### 常用命令总结

```dockerfile
docker build -t pflask:1.0	#创建镜像
docker image | grep python	#查看镜像
docker run --rm -p 5000:5000 pflask:1.0	#运行镜像
docker run --rm -p 5000:5000 -e env1=env1 pflask:1.0	#加入环境变量
docker run -d pflask:1.0	#运行docker并守护进程
docker ps -a	#查看当前正在运行的所有镜像
docker stop IMAGE_ID	#根据IMAGE_ID停止镜像进程
docker rm -f IMAGE_ID	#根据IMAGE_ID删除镜像进程
docker logs -f tname	#根据助记符查看日志
ss -anp | grep 5000		#从所有端口中过滤出5000
curl -v "http://192.168.195.100:5000"		#测试连接

```

## SSH基本用法

SSH之所以能够保证安全, 原因在于采用了公钥加密. 整个过程如下

1.   远程主机收到用户的登录请求, 把自己的公钥发给用户.
2.   用户使用这个公钥, 将登录密码加密后, 发送回来.
3.   远程主机用自己的私钥, 解密登录密码, 如果密码正确, 就同一用户登录.

![对称加密算法](Linux笔记记录.assets/download.png)

![非对称加密算法](Linux笔记记录.assets/download-16594481381123.png)

对称加密算法使用同一个密钥对数据进行加密和解密. SSH连接建立过程中生成的绘画密钥就是对称密钥, 该对称密钥是由客户端和服务器端基于共享的部分信息和各自的私有数据使用密钥交换算法分别生成. 因为对称加密算法解密速度很快, 所以适用于传输大量数据的场景.

非对称加密的发送和接收需要使用一对关联的SSH密钥, 公钥和私钥. 私钥由生成的一方自己保管, 公钥可以发送给任何请求通信的其他人. 发送放用收到的公钥对自己的同行内容进行加密, 只有接收方可以使用私钥进行解密获取同行内容. 非堆成加密的私钥不需要暴露在网络中, 但是加解密的速度必对称密钥慢得多.

SSH连接过程中的两个阶段使用了非对称加密, 一个是在密钥交换阶段, 服务器和客户端都生成了自己临时的公钥和私钥, 用于计算出同一个用于后续加密通信内容的会话密钥. 另一个就是在用户认证阶段, 利用只有匹配的私钥可以唯一解密公钥加密的内容这一特点, 通过客户端的公钥私钥对验证客户端的身份.

**密钥认证**
SSH用户认证最基本的两种方式密码认证和密钥认证. 密码认证是将自己的用户名和密码发送给服务器进行认证, 这种方式比较简单, 且每次登录都需要输入用户名和密码. 密钥认证使用公钥私钥对进行身份验证, 实现完全的免密登录, 这是一种广泛使用且推荐的登录方式. 密钥认证的基本原理是服务器使用客户端的公钥对随机内容加密, 客户端使用自己私钥解密并发送给服务器以证实自己的身份.

![SSH密钥认证登录流程](Linux笔记记录.assets/download-16594882773356.png)

SSH密钥认证登录流程

1.   在进行SSH连接之前, SSH客户端需要先生成自己的公钥私钥对, 并将自己的公钥放在SSH服务器上.
2.   SSH客户端发送登录请求, SSH服务器就会根据请求中的用户名等信息搜索客户端的公钥, 并用这个公钥加密一个随机数发送给客户端.
3.   客户端使用自己的私钥地返回信息进行解密, 并发送给服务器
4.   服务器验证客户端解密的信息是否正确, 如果正确, 则认证通过.

**免密登录**
执行`ssh-keygen`生成SSH钥匙:

```bash
local$ ssh-keygen
Generating public/private rsa key pair.
Enter file in which to save the key (/home/user/.ssh/id_rsa):
Created directory '/home/user/.ssh'.
Enter passphrase (empty for no passphrase):
Enter same passphrase again:
Your identification has been saved in /home/user/.ssh/id_rsa.
Your public key has been saved in /home/user/.ssh/id_rsa.pub.
The key fingerprint is:
SHA256:47VkvSjlFhKRgz/6RYdXM2EULtk9TQ65PDWJjYC5Jys user@local
The key's randomart image is:
+---[RSA 2048]----+
|       ...o...X+o|
|      . o+   B=Oo|
|       .....ooo*=|
|        o+ooo.+ .|
|       .SoXo.  . |
|      .E X.+ .   |
|       .+.= .    |
|        .o       |
|                 |
+----[SHA256]-----+
```

这段告诉我们, 生成的公钥放在了`~/.ssh/id_rsa.pub`, 私钥放在了`~/.ssh/id_rsa`, 接下来要让远程及其记住我们的公钥. 最简单的方法是`ssh-copy-id user@remote -p port`

```bash
local$ ssh-copy-id user@remote -p port
/usr/bin/ssh-copy-id: INFO: Source of key(s) to be installed: "/home/user/.ssh/id_rsa.pub"
/usr/bin/ssh-copy-id: INFO: attempting to log in with the new key(s), to filter out any that are already installed
/usr/bin/ssh-copy-id: INFO: 1 key(s) remain to be installed -- if you are prompted now it is to install the new keys
user@remote's password:

Number of key(s) added: 1

Now try logging into the machine, with:   "ssh 'user@remote' -p port"
and check to make sure that only the key(s) you wanted were added.
```

`ssh-copy-id`在绝大多数发行版上都有预安装, 在Mac上可以通过`brew install ssh-copy-id` 一键安装.

在没有`ssh-copy-id`的情况下, 也可以轻松做到这件事

```bash
ssh user@remote -p port 'mkdir -p .ssh && cat >> .ssh/authorized_keys' < ~/.ssh/id_rsa.pub
```

 这句话的以上是, 在远端执行新建`.ssh`文件, 并把本地的`~/.ssh/id_rsa.pub`追加到远端的`.ssh/authorized_keys`. 当然, 可以先复制公钥, 在登录远程机器, 粘贴到`~/.ssh/authorized_keys`中

**别名设置**
每次都输入`ssh user@remote -p port`, 时间就了不好记忆.

比如想用`ssh lab`来替代上面这么一长串, 那么在`~/.ssh/config`里追加一下内容

```bash
Host lab
	HostName remote
	User user
	Port port
```

保存之后, 即可用`ssh lab`登录, 如果还配置了公钥登入, 那就连密码都不用输入了.

**反向端口转发**
假设一台处于公网的机器jumpbox, 这台机器在任何地方都能范文; 而实验室也有一台机子lab, 这台机子只能在实验室内部访问, 但他可以访问公网, 我希望能在任何地方都访问这台机器. 使用`ssh -R`可以轻松做到这件事情

```bash
lab$ ssh -R 10022:localhost:22 jumpbox
jumpbox$ ssh user@localhost -p 10022
```

如果上面的过程成功了, 这说明在执行`ssh -R 10022:localhost:22 jumpbox`之后, 我成功地将lab上的22端口反向转发到了`jumpbox`的10022端口. 只要保持这个ssh不断, 任何一台机器可以首先连接到`jumpbox`, 然后通过`ssh user@localhost -p 10022`连回lab. 可以看到这里jumpbox起到了一个跳板的作用, 所以称作为跳板机.

```bash
ssh -NfR 10022:localhost:22 jumpbox
```

这句话里面`-N`表示不执行命令, 只做端口转发; `-f`表示在后台运行, 也就是说, 这句话执行之后`ssh`就在后台默默工作了. 

**正向端口转发**
反向端口转发是把本机的端口转发到远程的机子上; 与之对应, 正向端口转发则是把远程的端口转发到本地.

比如, 之前把lab的22端口反向转发到了jumpbox的10022端口上, 现在我想把它转发到本地机器的20022端口上, 只需要执行`ssh -L`就行了.

```bash
local$ ssh -NL 20022:localhost:10022 jumpbox
local$ ssh localhost -p 20022
```

**用作SOCKS5**代理

要是想要在家访问公司内网的一些网站, 但是公司又没有提供进入内网的VPN, 可通过`ssh -D`可以在本地建立起一个SOCK5代理

```
ssh -ND 1080 workplace
```

**ssh参数说明**

```bash
-c# 压缩传输, 提高传输速度
-f #将SSH传输转入后台执行, 不占用当前的shell
-N #建立静默连接(建立了连接, 但是看不到具体会话)
-g #运行远程主机连接本地用于转发的端口
-L #本地端口转发
-R #远程端口转发
-D #动态转发(socks代理)
-P #指定SSH端口
```



**cmd查看端口占用情况, 及对应进程, 杀进程**

地址一栏相关说明:

`0.0.0.0`是对外开放, 通过服务器域名, ip可以访问的端口

`127.0.0.1`是只对本机`localhost`访问, 也是保护此端口安全性

`netstat -ano`, 列出所用端口的情况. 在列表中观察被占用的端口, 比如49157.

>   -a: 显示所有连接和侦听端口, 服务器连接通常不显示
>
>   -e: 显示以太网统计, 该参数可以与-s选项结合使用
>
>   -n: 以数字格式显示地址和端口

![img](Linux笔记记录.assets/webp.webp)

2.   查看具体被占用的端口`netstat -ano | findstr "端口"`

     ![img](Linux笔记记录.assets/webp-165948959063911.webp)

3.   `tasklist | findstr PID`

     可以查询到当前占用这个端口的是那个应用

     ![img](Linux笔记记录.assets/1481828-20211022151805051-609094636.jpg)

4.   杀死进程

     `taskkill /pid PID -t -f`

     `-f`用来强制执行

在服务器中, 可以使用`netstat -tulup | grep ID` 查询是否建立连接

## SSH上传本地文件到linux

1. 从服务上下载文件

   `scp username@servername:/path/filename /var/www/local_dir(本地目录)`

   例如`scp root@192.168.0.101:/var/www/test.txt`#把192.168.0.101上的/var/www/test.txt下载到/var/www/local_dir(本地目录)

2. 上传本地文件到服务器

   `scp /path/filename username@servername:/path`

   例如`scp /var/www/test.txt root@192.168.0.101:/var/www`＃把本机/var/www/目录下的test.txt文件上传到192.168.0.101这台服务器上的/var/www/目录中

3. 从服务器下载整个目录

   `scp -r username@servername:/var/www/remote_dir(远程目录) /var/www/local_dir(本地目录)`

   例如`scp -r root@192.168.0.101:/var/www/test /var/www/`

## Error

1. 在ubuntu更新是, 出现`:`

   或者修改解析域名:`vi /etc/systemd/resolved.conf`

2. pip安装工具时报错提醒, Command "usr/bin/python -u -c" import setuptools, tokenize; `__file__` = "/tmppip-build-f8m_zq/statsmod_"

   原因是按照工具包时, 许哟啊抓取网页因而需要处理https, 而处理https又依赖加解密算法(即cryptography), 而cryptography又依赖傅里叶变换的算法以及相应的编译的环境. Unbutn 默认没有按照libffi-dev 和libssl-dev, gcc也不一定安装, 而目标安装包又没有将软件包寄到依赖列表里, 因此需要手动安装: 

   ```
   sudo apt-get install libssl-dev libffi-dev python-dev build-essential libxml2-dev libxslt1-dev
   ```
   
   **pycharm**
   
   -   远程连接失败
   
       ![image-20220802115033930](Linux笔记记录.assets/image-20220802115033930.png)
   
       Had the same bug here. SSH was reachable by mean of paaword in my terminal, but would’nt connect with Clion. “can’t connect to remote host: net.schmizz.sshj.transposrt.TransportException: Server closed connection during identification exchange”.
   
       To resolve this, I had to clear the content of `~/.ssh/config` which in my case was completely useless.
   
   3.   SSH隧道建立失败
   
        ![image-20220803094809858](Linux笔记记录.assets/image-20220803094809858.png)
   
        ```markdown
        1. `sssh-keygen -t rsa`在本机上生成密钥
        2. 将生成的`id_rsa.pub`上传到服务器`/~.ssh/authorized_keys`中
        3. 在本地建立隧道: `ssh -NfL localPort:remote_IP:remotePort user@IP -p port
        4. 检查隧道是否建立成功, 或者端口号是否被占用. `netstat -ano | findstr locaPort`
        ```
   
        
   
        

## reference

1. https://www.bookstack.cn/read/docker_practice-1.3.0/image-pull.md