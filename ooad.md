Effective programming 
========================

## S.O.L.I.D

### SRP (Single Responsibility)
- Class should be responsible for ONLY 1 job.
- _e.g._ - AreaCalculator and AreaOutputter should be different!

### OCP (Open Closed)
- __Open__ for addition / extending, __Closed__ for modification
- _NOTE_ : code to an interface

### LSP (Liskov substitution)
- Every subclass should be substitutable with their parent class.
- No change in params and return types enables this.
- LSP is needed where some code thinks it is calling the methods of a type T, 
and may unknowingly call the methods of a type S, where S extends T.
- applicable __ONLY__ when there is a family

### ISP (Interface segregation)
- Client should not be forced to implement an interface it doesnt use.
- _e.g._ - interface shouldnt be designed such that some client doesnt use all 
the methods! This implies interface responsibility needs to be segregated.

### DIP (Dependancy Inversion)
- High-level objects shouldnt depend on low-level objects, but should depend on abstractions!
- _NOTE_ : code to an interface

## Best practices
- __NO__ :
	- RTTI
	- bidirectional coupling
	- friend class
	- dynamic-cast / down-cast
	- multiple inheritance
	- DRY : Don't repeat yourself
	- private / protected inheritance
	- Interface-interface interaction! __No__ class-class interaction
	- singleton pattern
	- static / global variables
	- static methods
	- Arrow pattern
	- _*_ pointers (use smart pointers)
	- virtual inheritance


- __YES__ :
	- low coupling
	- Unit testable
		* data access		=	dAO (never depend on infrastucture)
		* method		=	interface / abstract
		* object instantiate	=	DI / factory
	- public / protected APIs must be virtual
	- pass objects by reference ( * / & )
	- virtual destructor


## Patterns

#### Adapter
Expose functionality through a wrapper to the outside world

#### Proxy
Wrap & enrich functionality

#### Decorator
Chain of enrichments


#### Strategy
replace portions of functionality using interface object

#### Template
normal overriding


#### Factory
provide instance of similar objects to outside world. Interface used by client

#### Abstract Factory
collection of factories

#### Observer
interface exposed to outside world, implemented by client
