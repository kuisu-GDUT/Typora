# MSYS安装

## MSYS介绍

it is a collection of tool and libraries providing you with an easy-to-use environment for building, installing and running native windows software. 

It consists of a command line terminal called [mintty](https://mintty.github.io/), bush, version control systems like git and subversion, tools like tar and awk and even build systems like autotools, all based on a modified version of [Cygwin](https://cygwin.com/). Despite some of these central parts being based on Cygwin, the main focus of MSYS2 is to provide a build environment for native windows software and the Cygwin-suing parts are kept at a minimum. MSYS2 provides up-to-date native builds for GCC, mingw-64, CPython, CMake, Meson, OpenSSL, FFmpeg, Rust, just to name a few.

To provide easy installation of packages and a way to keep them updated it features a package management system called [Pacman](https://wiki.archlinux.org/index.php/pacman), which should be familiar to Arch Linux users. It brings many powerful features such as dependency resolution and simple complete system upgrades, as well as straight-forward and reproducible package building.

## Installation

1. Download the installer: [msys2-x86_64-20210725.exe](https://github.com/msys2/msys2-installer/releases/download/2021-07-25/msys2-x86_64-20210725.exe) 
2. Run the installer. MSYS2 requires 64 bit windows or newer.
3. Enter your desired installation Folder (short ASCLL-only path on NTFS volum, no accents, no space, no symlinks, no subst or network drives, no FAT)

![img](.\msys install.assets\install-2-path.png)

4. When done, tick Run MSYS2 now.

![img](.\msys install.assets\install-3-finish.png)

5. Update the package database and base packages. Unless your setup file is very recent, it will take two steps. Firt run `pacman -Syu`

   ```shell
   $ pacman -Syu
   :: Synchronizing package databases...
    mingw32                        805.0 KiB
    mingw32.sig                    438.0   B
    mingw64                        807.9 KiB
    mingw64.sig                    438.0   B
    msys                           289.3 KiB
    msys.sig                       438.0   B
   :: Starting core system upgrade...
   warning: terminate other MSYS2 programs before proceeding
   resolving dependencies...
   looking for conflicting packages...
   
   Packages (6) bash-5.1.004-1  filesystem-2021.01-1
                mintty-1~3.4.4-1  msys2-runtime-3.1.7-4
                pacman-5.2.2-9  pacman-mirrors-20201208-1
   
   Total Download Size:   11.05 MiB
   Total Installed Size:  53.92 MiB
   Net Upgrade Size:      -1.24 MiB
   
   :: Proceed with installation? [Y/n]
   :: Retrieving packages...
    bash-5.1.004-1-x86_64            2.3 MiB
    filesystem-2021.01-1-any        33.2 KiB
    mintty-1~3.4.4-1-x86_64        767.2 KiB
    msys2-runtime-3.1.7-4-x86_64     2.6 MiB
    pacman-mirrors-20201208-1-any    3.8 KiB
    pacman-5.2.2-9-x86_64            5.4 MiB
   (6/6) checking keys in keyring       100%
   (6/6) checking package integrity     100%
   (6/6) loading package files          100%
   (6/6) checking for file conflicts    100%
   (6/6) checking available disk space  100%
   :: Processing package changes...
   (1/6) upgrading bash                 100%
   (2/6) upgrading filesystem           100%
   (3/6) upgrading mintty               100%
   (4/6) upgrading msys2-runtime        100%
   (5/6) upgrading pacman-mirrors       100%
   (6/6) upgrading pacman               100%
   :: To complete this update all MSYS2 processes including this terminal will be closed. Confirm to proceed [Y/n]
   
   ```

   6. Run "MSYS2 MSYS" from start menu. Update the rest of the base packages with `pacman -Su`

      ```shell
      $ pacman -Su
      :: Starting core system upgrade...
       there is nothing to do
      :: Starting full system upgrade...
      resolving dependencies...
      looking for conflicting packages...
      
      Packages (20) base-2020.12-1  bsdtar-3.5.0-1
                    [... more packages listed ...]
      
      Total Download Size:   12.82 MiB
      Total Installed Size:  44.25 MiB
      Net Upgrade Size:       3.01 MiB
      
      :: Proceed with installation? [Y/n]
      [... downloading and installation continues ...]
      ```

   7. Now MSYS2 is ready for you. You will probably want to intall some tools and the mingw-w64 GCC to start compiling:

      ```shell
      $ pacman -S --needed base-devel mingw-w64-x86_64-toolchain
      warning: file-5.39-2 is up to date -- skipping
      [... more warnings ...]
      :: There are 48 members in group base-devel:
      :: Repository msys
         1) asciidoc  2) autoconf  3) autoconf2.13  4) autogen
         [... more packages listed ...]
      
      Enter a selection (default=all):
      :: There are 19 members in group mingw-w64-x86_64-toolchain:
      :: Repository mingw64
         1) mingw-w64-x86_64-binutils  2) mingw-w64-x86_64-crt-git
         [... more packages listed ...]
      
      Enter a selection (default=all):
      resolving dependencies...
      looking for conflicting packages...
      
      Packages (123) docbook-xml-4.5-2  docbook-xsl-1.79.2-1
                     [... more packages listed ...]
                     m4-1.4.18-2  make-4.3-1  man-db-2.9.3-1
                     mingw-w64-x86_64-binutils-2.35.1-3
                     mingw-w64-x86_64-crt-git-9.0.0.6090.ad98746a-1
                     mingw-w64-x86_64-gcc-10.2.0-6
                     mingw-w64-x86_64-gcc-ada-10.2.0-6
                     mingw-w64-x86_64-gcc-fortran-10.2.0-6
                     mingw-w64-x86_64-gcc-libgfortran-10.2.0-6
                     mingw-w64-x86_64-gcc-libs-10.2.0-6
                     mingw-w64-x86_64-gcc-objc-10.2.0-6
                     mingw-w64-x86_64-gdb-10.1-2
                     mingw-w64-x86_64-gdb-multiarch-10.1-2
                    [... more packages listed ...]
      
      Total Download Size:    196.15 MiB
      Total Installed Size:  1254.96 MiB
      
      :: Proceed with installation? [Y/n]
      [... downloading and installation continues ...]
      ```

   8. To start building using the mingw-w64 GCC, close this window and run `MSYS MinGW 64-bit` from start menu. Now you can call `make` or `gcc` to build software for windows.

   9. Check out the introduction page to learn which start menu item to use when and which packages to install. Take look at [Detailed MSYS2 install guide](https://www.msys2.org/wiki/MSYS2-installation/) for troubleshooting and additional details on how to keep your MSYS2 up-to-date.

## MSYS2 install guid

They can be installed or extracted to any place on your computer, but you must use folder names that consist of only ascii characters and no space (also it make good sense to put it in a path that doesn't many characters due to PATH_MAX being 260; `c:msys64` is ideal)

The installer and base archives only contain the tools necessary to 1) start MSYS2 update the pre-installed packages and 3) install new packages.

- After installing or extracting MSYS2 you should start MSYS2 by executing msys2_shell.cmd
- Now we can update the base MSYS2 packages to their latest versions. 

Partial upgrades

1. run `pacman -Syuu`. Follow the instructions. Repeat this step until it syas there no package to update.
2. install new packages: `pacman -S <package_names|package_groups>`. For example, `pacman -S make gettext base-devel` In this example is a package group which contains many packages. 
3. Removing packages: `pacman -R <package_names|package_groups>`
4. Searching for packages: `pacman -Ss <name_pattern>`

## pacman

软件包管理器, 类似于ubuntu的apt-get一样

安装: `pacman -S package_name1 package_name2`

删除: `pacman -Rs package_name`

升级: `pacman -Su`

查询: `pacman -Ss package`

其他

```pacman
#仅下载包
pacman -Sw package_name
#安装本地包
pacman -U /path/to/package/package_name-version.pkg.tar.gz
#安装一个远程包
pacman -U http://url/package_name-version.pkg.tar.gzs
```

## reference

- [MSYS2-Installation - MSYS2](https://www.msys2.org/wiki/MSYS2-installation/)