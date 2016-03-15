# <center>Building kernel modules</center>

### Using kbuild
- `obj-m` : name of the module to build
- `<name>-y` : list of the dependencies
- `ccflags-y` : list of headers 
- KDIR : kernel headers, should be set to `/lib/modules/`uname -r`/build`
- PWD : location of the kbuild / Makefile file


### Sample Makefile
```
ifneq ($(KERNELRELEASE),)
	# kbuild part of makefile
	obj-m  := complex.o
	# complex-y := src/complex_main.o test.o
	#ccflags-y	:=	$(src)/include
	
else
	# normal makefile
	KDIR ?= /lib/modules/`uname -r`/build

	default:
		$(MAKE) -C $(KDIR) M=$(PWD) modules
		#$(MAKE) -C $(KDIR) M=$(PWD) modules_install

	# Module specific targets
	#genbin:
	#	echo "X" > complex_bin.o_shipped

endif
```

[kbuild documentation](http://www.mjmwired.net/kernel/Documentation/kbuild/modules.txt)
