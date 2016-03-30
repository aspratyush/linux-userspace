<center>Linux Sysdmin</center>
===============================

### Table of Contents
1. [Linux Process](#linuxProcess)
   * [PID](#pid)
   * [UID and EUID](#uid-euid)
   * [Niceness](#niceness)
   * [Process States](#process-states)
   * [Signals](#signals)
2. [Kernel Scheduling](#kernel-sched)
   * [Proc FS](#procfs)
   * [strace](#strace)


## 1. Linux Process <a name="linuxProcess"></a>
Two parts:

- **Address space** : physical memory address (RAM) that the process maps to.
  * Processes sees the virtual memory provided by the kernel

- **Kernel Data Structures** : all information dealing with the process that the 
kernel needs to know.
  * owner, parent, address space
  * resources, files and N/W ports access, signal components


### 1.1 PID<a name="pid"></a>
Each process gets a process ID
- PID1 - 1st process that gets spawned by kernel
- next available PID allotted
- each process spawned by a parent
- if parent dies, child gets reparented to PID1

### 1.2 UID and EUID<a name="uid-euid"></a>
Effective-UID : do not pass permissions of user to the app

### 1.3 NICENESS<a name="niceness"></a>
How "nice" a process is being to others, i.e, much resources a process can hog
- Low : less nice.. higher priority
- High : Nicer the process.. low priority task
- For starting a process with specific nice value:  
```bash
$ nice -n {-20,19} <processName>
```
- NICEVAL = -20 will ensure process takes up ALL resource
- For changing nice value of an existing process:  
```bash
$ sudo renice <niceVal> <PID>
```

### 1.4 FORK<a name="fork"></a>
Parent process forks to create the child

### 1.5 STATES<a name="process-states"></a>
- runnable : currently running 
- sleeping : waiting for something
- zombie : finished its work, waiting to return data
- stopped : process that has received SIGSTOP. Needs SIGCONT to resume.

## 2. SIGNALS<a name="signals"></a>
- How process communicate with each other / with kernel
- notify parent about child exiting
- H/W addition by kernel to processes
- Some standard SIGNALS ```man signal```
- Process can handle the signal using ```sighandler / sigaction```
```bash
$ kill -l # list the signals  
$ sudo killall <processName> # kill all process started by process   
```

## 3. Kernel Scheduling<a name="kernel-sched"></a>
- Every process wants CPU time.. Kernel needs to schedule the processes.
 
### 3.1 Proc FS <a name=procfs></a>
- /proc is a virtual FS where kernel does book-keeping for all processes 
- kernel adds info about all running process here, ordered by **PID**
- folders are empty, since info is updated at runtime
```
- cmd # what command is being run  
- cwd # where is it operating from  
- environ # environment variables used  
- maps # memory mapping info  
- statm # memory info  
```

### 3.2 strace<a name="strace"></a>
Attach/Dettach to a process


## 4. Filesystem
-----------------


```$ man hier``` gives info of the filesystem

* ```/boot``` : bootloader and kernel image
* ```/dev``` : devices mount point
* ```/etc``` : config files for processes
* ```/sbin```  : secure system binaries
* ```/bin``` : soft links to actual binaries
* ```/lib``` : shared libraries
* ```/mnt``` : external storage mount point
* ```/opt``` : softwares other than the base system
* ```/proc``` : process management virtual FS, manages process states
* ```/var``` : various things, such as logs

File types (7 types):
* directory : contain files and sub-directories
* files
* symbolic links
* block device : storage devices are seen as block devices (kernel buffers these devices)
* character device : some devices (mouse, etc.) seen as character devices (direct access to the device)
* unix-sockets : communciation channel between processes
* named-pipes : FIFO communciation channel
